"""SawSeenVLA-Qwen pre/post-processor pipeline.

Differs from SawSeenVLA in that we DO NOT tokenize text in the data pipeline.
Qwen needs (text + images) tokenized together via its AutoProcessor — that happens
inside the model's `forward` to keep the data pipeline schema-stable.

The data pipeline therefore only:
  1. Renames features (compat)
  2. Adds a batch dimension
  3. Moves to device
  4. Normalizes state/action via the standard NormalizerProcessorStep

Images stay as float [0, 1] tensors (no Qwen-side preprocessing here); the encoder
inside the model converts to uint8 [0, 255] before calling Qwen's processor.
"""
from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_sawseenvla_qwen import SawSeenVLAQwenConfig


def make_sawseenvla_qwen_pre_post_processors(
    config: SawSeenVLAQwenConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
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
