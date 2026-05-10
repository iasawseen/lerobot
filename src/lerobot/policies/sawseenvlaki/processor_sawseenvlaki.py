#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_sawseenvlaki import SawSeenVLAKIConfig


@ProcessorStepRegistry.register(name="sawseenvlaki_fast_action_tokenizer_processor_step")
@dataclass
class SawSeenVLAKIFastActionTokenizerProcessorStep(ProcessorStep):
    """Pre-tokenize action chunks into raw FAST IDs (0..K-1).

    Mirrors :class:`lerobot.processor.ActionTokenizerProcessorStep` but
    skips its PaliGemma-specific vocab remap and BOS/"Action:"/"|"
    wrapping. SawSeenVLAKI consumes raw FAST IDs through a dedicated
    ``fast_embed`` table (separate from the SmolVLM2 vocab), so the
    tokens shipped in the batch must stay in ``[0, fast_vocab_size)``.

    Output (in ``complementary_data``):
      - ``ACTION_TOKENS``     : LongTensor [B, max_action_tokens]
      - ``ACTION_TOKEN_MASK`` : BoolTensor [B, max_action_tokens]
                                (True for real tokens, False for padding)

    During inference the action key is absent from the transition; we
    skip silently.
    """

    fast_action_tokenizer_path: str = "lerobot/fast-action-tokenizer"
    max_action_tokens: int = 32
    trust_remote_code: bool = True
    pad_token_id: int = 0
    # Filled in by ``__post_init__``; not a config field.
    action_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        try:
            from transformers import AutoProcessor
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'transformers' library is required for "
                "SawSeenVLAKIFastActionTokenizerProcessorStep."
            ) from exc

        self.action_tokenizer = AutoProcessor.from_pretrained(
            self.fast_action_tokenizer_path,
            trust_remote_code=self.trust_remote_code,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        tokens, mask = self._tokenize_action(action)

        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        complementary_data[ACTION_TOKENS] = tokens
        complementary_data[ACTION_TOKEN_MASK] = mask
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def _tokenize_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize action chunk into FAST IDs, right-padded to ``max_action_tokens``.

        Args:
            action: shape (B, H, action_dim) or (H, action_dim).

        Returns:
            tokens: LongTensor (B, max_action_tokens) — FAST IDs in [0, fast_vocab_size).
            mask:   BoolTensor (B, max_action_tokens) — True for real tokens.
        """
        single_sample = action.dim() == 2
        if single_sample:
            action = action.unsqueeze(0)

        batch_size = action.shape[0]
        device = action.device
        max_len = self.max_action_tokens

        tokens_list: list[torch.Tensor] = []
        masks_list: list[torch.Tensor] = []
        for i in range(batch_size):
            action_cpu = action[i : i + 1].cpu()
            ids = self.action_tokenizer(action_cpu)
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids, dtype=torch.long, device=device)
            else:
                ids = ids.to(device=device, dtype=torch.long)
            if ids.dim() > 1:
                ids = ids.flatten()

            if ids.numel() > max_len:
                logging.warning(
                    "FAST token length (%d) exceeds max_action_tokens (%d); truncating. "
                    "Consider increasing fast_max_action_tokens.",
                    ids.numel(),
                    max_len,
                )
                ids = ids[:max_len]
                m = torch.ones(max_len, dtype=torch.bool, device=device)
            else:
                pad_n = max_len - ids.numel()
                m = torch.cat(
                    [
                        torch.ones(ids.numel(), dtype=torch.bool, device=device),
                        torch.zeros(pad_n, dtype=torch.bool, device=device),
                    ]
                )
                ids = torch.nn.functional.pad(ids, (0, pad_n), value=self.pad_token_id)

            tokens_list.append(ids)
            masks_list.append(m)

        tokens = torch.stack(tokens_list, dim=0)
        mask = torch.stack(masks_list, dim=0)
        if single_sample:
            tokens = tokens.squeeze(0)
            mask = mask.squeeze(0)
        return tokens, mask

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_sawseenvlaki_pre_post_processors(
    config: SawSeenVLAKIConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Pre-/post-processor pipelines for SawSeenVLAKI.

    With ``ki_enabled=False`` (default), this matches the SawSeenVLA
    pipeline exactly. When ``ki_enabled=True``, an additional
    ``SawSeenVLAKIFastActionTokenizerProcessorStep`` runs after
    normalization to pre-tokenize the action chunk into raw FAST IDs
    that the model's forward consumes alongside the continuous actions.
    """

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    if config.ki_enabled:
        input_steps.append(
            SawSeenVLAKIFastActionTokenizerProcessorStep(
                fast_action_tokenizer_path=config.fast_action_tokenizer_path,
                max_action_tokens=config.fast_max_action_tokens,
            )
        )

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
