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

"""
SawSeenVLAWM:

Structural clone of SmolVLA. Same VLM-backbone-with-action-expert architecture,
registered separately so it can be fine-tuned and iterated on without affecting
the upstream SmolVLA policy.

Install smolvla extra dependencies (the SmolVLM2 backbone is shared):
```bash
pip install -e ".[smolvla]"
```

Example of training SawSeenVLAWM from a fresh action expert (loads only the
SmolVLM2 backbone weights):
```bash
lerobot-train \
--policy.type=sawseenvlawm \
--dataset.repo_id=<USER>/<dataset> \
--batch_size=64 \
--steps=200000
```

Example of using a trained SawSeenVLAWM checkpoint:
```python
policy = SawSeenVLAWMPolicy.from_pretrained("<USER>/<sawseenvlawm_checkpoint>")
```
"""

import math
from collections import deque
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.device_utils import get_safe_dtype
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from ..rtc.modeling_rtc import RTCProcessor
# The SmolVLM2 backbone wrapper is generic and shared with SmolVLA.
from ..smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from .smolvlm_with_two_experts import SmolVLMWithTwoExpertsModel
from ..utils import (
    populate_queues,
)
from .configuration_sawseenvlawm import SawSeenVLAWMConfig
from .lewm_encoder import LeWMVisionEncoder, LeWMWorldModel


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with sawseenvlawm which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by sawseenvlawm to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SawSeenVLAWMPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SawSeenVLAWMConfig
    name = "sawseenvlawm"

    def __init__(
        self,
        config: SawSeenVLAWMConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        require_package("transformers", extra="smolvla")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")
        loss_dict = {}

        chunk_end_images = None
        chunk_end_pad_mask = None
        if self.config.latent_goal_enabled:
            chunk_end_images, chunk_end_pad_mask = self.prepare_chunk_end_images(batch)

        losses, latent_goal_loss = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time,
            chunk_end_images=chunk_end_images,
            chunk_end_pad_mask=chunk_end_pad_mask,
        )
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over valid (time, action) entries
            if actions_is_pad is None:
                per_sample_loss = losses.mean(dim=(1, 2))
            else:
                num_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
                per_sample_loss = losses.sum(dim=(1, 2)) / num_valid
            # Latent Goal Expert loss is a scalar; broadcast it to per-sample for the
            # weighted-sum convention used by sample_weighter consumers.
            if latent_goal_loss is not None:
                per_sample_loss = per_sample_loss + self.config.latent_goal_loss_weight * latent_goal_loss
                loss_dict["loss_latent_goal"] = latent_goal_loss.item()
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss over valid (time, action) entries
            if actions_is_pad is None:
                loss_action = losses.mean()
            else:
                num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
                loss_action = losses.sum() / num_valid
            loss_dict["loss_action"] = loss_action.item()
            if latent_goal_loss is not None:
                loss = loss_action + self.config.latent_goal_loss_weight * latent_goal_loss
                loss_dict["loss_latent_goal"] = latent_goal_loss.item()
            else:
                loss = loss_action
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SawSeenVLAWM preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch.
        # When ``observation_delta_indices = [0, chunk_size]`` (Latent Goal Expert
        # enabled) the obs stack is shape (B, 2, C, H, W) with index 0 the
        # anchor frame and index 1 the chunk-end frame. We always take the
        # anchor at index 0 here; ``prepare_chunk_end_images`` reads index 1
        # separately for the Latent Goal Expert regression target.
        for key in present_img_keys:
            img = batch[key][:, 0, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_chunk_end_images(self, batch):
        """Extract and preprocess the chunk-end frame for Latent Goal Expert.

        Mirrors ``prepare_images`` but reads index 1 (the o_{t+chunk_size}
        frame) and additionally returns a per-sample boolean mask marking
        samples whose chunk-end fell past the episode boundary (so the
        dataset returned padded frames). The Latent Goal Expert loss masks those out.

        Returns ``(images, chunk_end_pad_mask)`` where ``chunk_end_pad_mask``
        has shape ``(B,)`` and is True for samples to drop from the Latent Goal Expert loss.
        """
        images = []
        per_camera_pads = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                "No image features in the batch — cannot extract chunk-end frame for Latent Goal Expert."
            )
        for key in present_img_keys:
            if batch[key].ndim != 5 or batch[key].shape[1] < 2:
                raise ValueError(
                    f"Latent Goal Expert expects observation stack shape (B, 2, C, H, W) for "
                    f"{key}, got {tuple(batch[key].shape)}. Confirm "
                    f"observation_delta_indices=[0, chunk_size]."
                )
            img = batch[key][:, 1, :, :, :]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            img = img * 2.0 - 1.0
            images.append(img)
            is_pad_key = f"{key}_is_pad"
            if is_pad_key in batch:
                per_camera_pads.append(batch[is_pad_key][:, 1])

        chunk_end_pad_mask = None
        if per_camera_pads:
            chunk_end_pad_mask = per_camera_pads[0]
            for p in per_camera_pads[1:]:
                chunk_end_pad_mask = chunk_end_pad_mask | p
        return images, chunk_end_pad_mask

    def prepare_state(self, batch):
        """Pad state. Index 0 is the anchor frame; Latent Goal Expert-mode chunk-end state
        at index 1 is intentionally discarded — it's the *outcome* of the
        actions and isn't an input."""
        state = batch[OBS_STATE][:, 0, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        """LoRA adapters on the frozen SmolVLM2 text_model q/v_proj only.

        The action expert and (when enabled) the Latent Goal Expert are
        randomly initialized — LoRA-wrapping them would freeze random
        weights. They go to ``modules_to_save`` instead, alongside the
        small projections (state, action in/out, time MLP, le-wm,
        Latent Goal Expert, Mode 3 inject). Vision encoder is left out
        of LoRA targeting so its activations aren't retained during
        backward (same blow-up sawseenvla hit at bs=64).

        Gradients from both experts still reach the LoRA adapters: the
        wrapper stores prefix K/V in ``past_key_values`` without
        ``.detach()``, so each expert's cross-attention into the cache
        is autograd-connected back to ``vlm.text_model.*.{q,v}_proj``
        and through them into the LoRA A/B matrices.
        """
        target_modules = r"model\.vlm_with_expert\.vlm\.model\.text_model\..*\.self_attn\.(q|v)_proj"

        modules_to_save = [
            "lm_expert",
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
        cfg = self.config
        if cfg.lewm_encoder_path is not None and cfg.lewm_inject_to == "suffix":
            modules_to_save.append("lewm_proj")
        if cfg.latent_goal_enabled:
            modules_to_save += [
                "latent_goal_expert",
                "latent_goal_in_proj",
                "latent_goal_out_proj",
                "latent_goal_time_mlp_in",
                "latent_goal_time_mlp_out",
                "latent_goal_anchor_proj",
            ]
        if cfg.latent_goal_inject_to_action:
            modules_to_save += [
                "latent_goal_action_zt_proj",
                "latent_goal_action_zg_proj",
            ]
        return {
            "target_modules": target_modules,
            "modules_to_save": modules_to_save,
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SawSeenVLAWM.

        Skips the base-class ``pretrained_path`` requirement: for
        SawSeenVLAWM the LoRA targets the **VLM** (which is loaded via
        ``load_vlm_weights=True`` from the HF Hub, not from a
        SawSeenVLAWM checkpoint). Either a SawSeenVLAWM pretrained_path
        OR load_vlm_weights=True is enough to make PEFT meaningful.
        """
        if not self.config.pretrained_path and not self.config.load_vlm_weights:
            raise ValueError(
                "PEFT is enabled but neither pretrained_path nor "
                "load_vlm_weights is set. LoRA targets the frozen pretrained "
                "VLM; with random VLM init there is nothing useful to adapt. "
                "Set load_vlm_weights=True (to fine-tune from the HF VLM "
                "checkpoint) or supply --policy.path=<sawseenvlawm_ckpt> (to "
                "adapt an existing SawSeenVLAWM checkpoint), or disable PEFT."
            )


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SawSeenVLAWM flow-matching action expert.

    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SawSeenVLAWMConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        # When Latent Goal Expert is enabled, the wrapper holds a second flow-matching
        # expert (latent_goal_expert) parallel to the action expert. Both experts share
        # only the VLM backbone — separate weights, separate projections, same
        # per-layer interleaving as today's single-expert path.
        wrapper_kwargs = dict(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        if self.config.latent_goal_enabled:
            self.vlm_with_expert = SmolVLMWithTwoExpertsModel(
                **wrapper_kwargs,
                latent_goal_expert_width_multiplier=self.config.latent_goal_expert_width_multiplier,
                latent_goal_num_expert_layers=self.config.latent_goal_num_expert_layers,
            )
        else:
            self.vlm_with_expert = SmolVLMWithExpertModel(**wrapper_kwargs)
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        # Latent Goal Expert projections — only built when the flag is on. The Latent Goal Expert
        # expert emits a 192-dim velocity (le-wm latent dim), so its in/out
        # projections sit between 192 and the Latent Goal Expert's hidden size.
        # ``latent_goal_anchor_proj`` projects the *current* le-wm state z_t into the
        # Latent Goal Expert's input space; it sits next to the denoising token in the
        # Latent Goal Expert suffix so the expert sees both "where am I now" and "noisy
        # future to denoise." Phase A keeps both at single-token granularity
        # (CLS-only); ``lewm_num_tokens`` doesn't widen the Latent Goal Expert anchor here.
        self.latent_goal_in_proj: nn.Linear | None = None
        self.latent_goal_out_proj: nn.Linear | None = None
        self.latent_goal_time_mlp_in: nn.Linear | None = None
        self.latent_goal_time_mlp_out: nn.Linear | None = None
        self.latent_goal_anchor_proj: nn.Linear | None = None
        # Mode 3 projections: route z_t (current le-wm CLS) and z_g (Latent Goal Expert
        # predicted or encoded chunk-end CLS) into the action expert's
        # suffix hidden space as two prepended tokens. Built only when the
        # flag is on; left as None otherwise so the absence of the
        # parameters is the test for "do we inject."
        self.latent_goal_action_zt_proj: nn.Linear | None = None
        self.latent_goal_action_zg_proj: nn.Linear | None = None
        if self.config.latent_goal_enabled:
            latent_goal_hidden = self.vlm_with_expert.latent_goal_expert_hidden_size
            # le-wm encoder output dim is 192 (ViT-Tiny). Hard-coded here
            # because the encoder isn't loaded yet at projection-init time
            # when ``lewm_encoder_path`` is None — but Latent Goal Expert always
            # targets le-wm space regardless of whether action-expert
            # injection is on.
            latent_goal_latent_dim = 192
            self.latent_goal_in_proj = nn.Linear(latent_goal_latent_dim, latent_goal_hidden)
            self.latent_goal_out_proj = nn.Linear(latent_goal_hidden, latent_goal_latent_dim)
            self.latent_goal_time_mlp_in = nn.Linear(latent_goal_hidden * 2, latent_goal_hidden)
            self.latent_goal_time_mlp_out = nn.Linear(latent_goal_hidden, latent_goal_hidden)
            self.latent_goal_anchor_proj = nn.Linear(latent_goal_latent_dim, latent_goal_hidden)
            if self.config.latent_goal_inject_to_action:
                action_hidden = self.vlm_with_expert.expert_hidden_size
                self.latent_goal_action_zt_proj = nn.Linear(latent_goal_latent_dim, action_hidden)
                self.latent_goal_action_zg_proj = nn.Linear(latent_goal_latent_dim, action_hidden)

        # le-wm visual side-channel (frozen ViT-Tiny trained on Libero).
        # When ``lewm_encoder_path`` is set, le-wm tokens are prepended to
        # the action expert's suffix (``lewm_inject_to="suffix"``). The
        # alternative value ``"none"`` loads the encoder but skips
        # action-expert injection — used by Latent Goal Expert to consume the
        # encoder's raw 192-dim features without contaminating the action
        # stream.
        self.lewm_encoder: LeWMVisionEncoder | None = None
        self.lewm_proj: nn.Linear | None = None
        if self.config.lewm_encoder_path is not None:
            if self.config.lewm_inject_to not in ("suffix", "none"):
                raise ValueError(
                    f"lewm_inject_to must be 'suffix' or 'none'; got {self.config.lewm_inject_to!r}"
                )
            self.lewm_encoder = LeWMVisionEncoder.from_lewm_checkpoint(
                self.config.lewm_encoder_path,
                num_tokens=self.config.lewm_num_tokens,
                image_height=self.config.lewm_image_height,
                image_width=self.config.lewm_image_width,
                patch_size=self.config.lewm_patch_size,
                freeze=self.config.lewm_freeze,
            )
            # Only build the action-expert projection when we're actually
            # injecting tokens into the action expert. ``"none"`` leaves
            # ``lewm_proj`` as None so ``compute_lewm_tokens`` returns None
            # and the suffix path skips injection naturally.
            if self.config.lewm_inject_to == "suffix":
                proj_out = self.vlm_with_expert.expert_hidden_size
                self.lewm_proj = nn.Linear(self.lewm_encoder.output_dim, proj_out)

        if self.config.latent_goal_enabled and self.lewm_encoder is None:
            raise ValueError(
                "latent_goal_enabled=True requires lewm_encoder_path to be set — "
                "Latent Goal Expert regresses against the le-wm encoded chunk-end frame, "
                "which is only available when the encoder is loaded."
            )

        # Phase B / MPC: full le-wm JEPA (encoder + projector +
        # action_encoder + predictor + pred_proj). Loaded only when MPC
        # is enabled — the training loss path never touches this module.
        # The pickle is the same as ``lewm_encoder_path`` (the encoder
        # we already loaded above is one of its sub-modules), so we can
        # fall back to that path when ``mpc_predictor_path`` is unset.
        self.lewm_world: LeWMWorldModel | None = None
        if self.config.mpc_enabled:
            predictor_path = self.config.mpc_predictor_path or self.config.lewm_encoder_path
            self.lewm_world = LeWMWorldModel.from_lewm_checkpoint(
                predictor_path,
                num_tokens=self.config.lewm_num_tokens,
                image_height=self.config.lewm_image_height,
                image_width=self.config.lewm_image_width,
                patch_size=self.config.lewm_patch_size,
                freeze=True,
            )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])

        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def compute_lewm_tokens(self, images: list[torch.Tensor]) -> torch.Tensor | None:
        """Concatenate cameras horizontally and run the le-wm encoder once.

        le-wm trained on libero with the two cameras (agentview +
        eye-in-hand) stacked horizontally as a single ``(H, 2W, 3)``
        input. To stay in-distribution we replicate that here: per-camera
        tensors from ``prepare_images`` are concatenated along the W axis
        before the encoder forward. The encoder bilinearly resizes the
        result to ``(image_height, image_width)`` (defaults 224×448),
        applies ImageNet normalization, and slices ``num_tokens`` from the
        ViT output.

        Returns a ``(B, num_tokens, expert_hidden_size)`` tensor or None
        if the encoder is disabled. Called once per
        ``forward``/``sample_actions`` and reused across denoising steps.
        """
        if self.lewm_encoder is None or self.lewm_proj is None:
            return None
        # Concatenate cameras along the width axis. Order follows
        # prepare_images' iteration over present_img_keys.
        if len(images) == 1:
            stacked = images[0]
        else:
            stacked = torch.cat(images, dim=-1)
        tokens = self.lewm_encoder(stacked)  # (B, num_tokens, 192)
        return self.lewm_proj(tokens.to(self.lewm_proj.weight.dtype))

    def embed_suffix(
        self,
        noisy_actions,
        timestep,
        lewm_tokens: torch.Tensor | None = None,
        latent_goal_inject_tokens: torch.Tensor | None = None,
    ):
        """Embed noisy_actions and timestep (plus optional le-wm and Latent Goal Expert
        injection tokens) for the action expert.

        ``latent_goal_inject_tokens`` (Mode 3) is a (B, 2, expert_hidden) tensor of
        already-projected [z_t, z_g] tokens. It sits at the very front of
        the suffix with ``att_mask=[1, 0]`` (one bidirectional block of two
        tokens), before any optional ``lewm_tokens`` and the causal action
        chunk. Cumsum ordering: action tokens see [z_t, z_g, lewm, prior
        actions]; z_t / z_g themselves see only the prefix.

        ``lewm_tokens`` is prepended so the (causal) action tokens can each
        attend to all le-wm tokens. The le-wm block itself uses
        ``att_mask=[1, 0, 0, ..., 0]`` so all of its tokens share one
        bidirectional attention block.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype

        # Optional Mode 3 Latent Goal Expert injection tokens — at the very front so they
        # are conditioning, not conditioned. Bidirectional within their
        # 2-token block.
        if latent_goal_inject_tokens is not None:
            latent_goal_inject_tokens = latent_goal_inject_tokens.to(dtype=dtype)
            embs.append(latent_goal_inject_tokens)
            num_inject = latent_goal_inject_tokens.shape[1]
            inject_mask = torch.ones(bsize, num_inject, dtype=torch.bool, device=device)
            pad_masks.append(inject_mask)
            att_masks += [1] + [0] * (num_inject - 1)

        # Optional le-wm tokens — prepended.
        if lewm_tokens is not None:
            lewm_tokens = lewm_tokens.to(dtype=dtype)
            embs.append(lewm_tokens)
            num_lewm = lewm_tokens.shape[1]
            lewm_mask = torch.ones(bsize, num_lewm, dtype=torch.bool, device=device)
            pad_masks.append(lewm_mask)
            att_masks += [1] + [0] * (num_lewm - 1)

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def embed_latent_goal_suffix(
        self,
        anchor_z: torch.Tensor,
        noisy_z: torch.Tensor,
        timestep: torch.Tensor,
    ):
        """Build the Latent Goal Expert's 2-token suffix.

        Tokens (in this order):
          0. ``anchor_z_t`` — frozen le-wm encoding of the *current* frame,
             projected into Latent Goal Expert space. Acts as the same-space anchor
             the Latent Goal Expert conditions on (matches Phase B's WM rollout
             starting point).
          1. ``noisy_z_g + time`` — the denoising token; flow-matching
             velocity is read from this position at the Latent Goal Expert's output.

        Both tokens share one attention block (att_mask=[1, 0]) so they're
        bidirectional within the Latent Goal Expert suffix. The block sits after the action
        tokens in the global cumulative-mask scheme, so action tokens cannot
        see Latent Goal Expert. The reverse direction (Latent Goal Expert → action / Latent Goal Expert → suffix-lewm) is
        blocked by an additional 2D-mask edit in ``forward()`` so that Latent Goal Expert
        predicts the chunk-end state purely from (prefix, z_t anchor),
        independent of the action chunk being committed.
        """
        if anchor_z.dim() == 2:
            anchor_z = anchor_z.unsqueeze(1)
        if noisy_z.dim() == 2:
            noisy_z = noisy_z.unsqueeze(1)

        # Anchor token: deterministic projection of z_t into Latent Goal Expert hidden.
        anchor_emb = self.latent_goal_anchor_proj(
            anchor_z.to(self.latent_goal_anchor_proj.weight.dtype)
        )  # (B, 1, latent_goal_hidden)

        # Denoising token: fuse noisy z with sin-cos time embedding.
        z_emb = self.latent_goal_in_proj(noisy_z.to(self.latent_goal_in_proj.weight.dtype))
        bsize = z_emb.shape[0]
        device = z_emb.device
        dtype = z_emb.dtype

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.latent_goal_expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        ).type(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(z_emb)

        z_time_emb = torch.cat([z_emb, time_emb], dim=2)
        z_time_emb = self.latent_goal_time_mlp_in(z_time_emb)
        z_time_emb = F.silu(z_time_emb)
        z_time_emb = self.latent_goal_time_mlp_out(z_time_emb)

        # Concat anchor + denoising token into a 2-token Latent Goal Expert suffix.
        anchor_emb = anchor_emb.to(dtype=z_time_emb.dtype)
        embs = torch.cat([anchor_emb, z_time_emb], dim=1)  # (B, 2, latent_goal_hidden)

        pad_mask = torch.ones(bsize, 2, dtype=torch.bool, device=device)
        # att_mask=[1, 0]: anchor opens a new attention block, denoising
        # token shares it (same cumsum → bidirectional within Latent Goal Expert).
        att_mask = torch.tensor([1.0, 0.0], dtype=z_time_emb.dtype, device=device)
        att_mask = att_mask[None, :].expand(bsize, 2)
        return embs, pad_mask, att_mask

    def _encode_lewm_cls(self, images: list[torch.Tensor]) -> torch.Tensor:
        """Encode camera-concatenated images via the frozen le-wm encoder
        and return the CLS token (B, 192).

        Used for both the Latent Goal Expert anchor (current frame) and the Latent Goal Expert regression
        target (chunk-end frame). Same camera-concat / 224×448 prep that
        the side-channel encoder uses; gradients are blocked because the
        encoder is frozen and we only train the Latent Goal Expert head.
        """
        if len(images) == 1:
            stacked = images[0]
        else:
            stacked = torch.cat(images, dim=-1)
        with torch.no_grad():
            tokens = self.lewm_encoder(stacked)  # (B, num_tokens, 192)
        return tokens[:, 0, :]  # CLS, (B, 192)

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None,
        chunk_end_images: list[torch.Tensor] | None = None,
        chunk_end_pad_mask: torch.Tensor | None = None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        suffix_lewm = self.compute_lewm_tokens(images)  # None unless lewm_inject_to="suffix"
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        latent_goal_loss: torch.Tensor | None = None
        prefix_len = prefix_pad_masks.shape[1]

        if self.config.latent_goal_enabled and self.config.latent_goal_inject_to_action:
            # ─────────────────────────── Mode 3 ───────────────────────────
            # Sequential 3-pass forward: the action expert reads LGE's
            # output (predicted z_g) as an extra suffix token, so the two
            # experts can no longer run in parallel through the shared
            # backbone. Pass 1 fills the VLM K/V cache once (prefix only);
            # Pass 2 runs the LGE reading that cache and emits its
            # velocity for the LGE flow-matching loss; Pass 3 runs the
            # action expert reading the same cache, with [z_t, z_g]
            # prepended to its suffix. Cache is read-only in Passes 2/3
            # (``fill_kv_cache=False``) so action-expert and LGE never see
            # each other's K/V.
            if chunk_end_images is None:
                raise ValueError(
                    "latent_goal_inject_to_action=True requires chunk_end_images in "
                    "forward(); observation_delta_indices=[0, chunk_size] should be set."
                )
            z_t_anchor = self._encode_lewm_cls(images)              # (B, 192)
            z_g_target = self._encode_lewm_cls(chunk_end_images)    # (B, 192)

            # ── Pass 1: prefix-only forward → VLM K/V cache ──
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, past_key_values = self.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None, None],
                use_cache=True,
                fill_kv_cache=True,
            )

            # ── Pass 2: LGE forward → flow-matching velocity & loss ──
            latent_goal_noise = self.sample_noise(z_g_target.shape, z_g_target.device)
            latent_goal_time = self.sample_time(z_g_target.shape[0], z_g_target.device)
            latent_goal_t_exp = latent_goal_time[:, None]
            latent_goal_x_t = (
                latent_goal_t_exp * latent_goal_noise + (1 - latent_goal_t_exp) * z_g_target
            )
            latent_goal_u_t = latent_goal_noise - z_g_target

            latent_goal_embs, latent_goal_pad_masks, latent_goal_att_masks = (
                self.embed_latent_goal_suffix(z_t_anchor, latent_goal_x_t, latent_goal_time)
            )
            pass2_pad = torch.cat([prefix_pad_masks, latent_goal_pad_masks], dim=1)
            pass2_att = torch.cat([prefix_att_masks, latent_goal_att_masks], dim=1)
            pass2_2d = make_att_2d_masks(pass2_pad, pass2_att)
            latent_goal_attention_mask = pass2_2d[:, prefix_len:, :]
            pass2_position_ids_full = torch.cumsum(pass2_pad, dim=1) - 1
            latent_goal_position_ids = pass2_position_ids_full[:, prefix_len:]

            outputs2, _ = self.vlm_with_expert.forward(
                attention_mask=latent_goal_attention_mask,
                position_ids=latent_goal_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, None, latent_goal_embs],
                use_cache=self.config.use_cache,
                fill_kv_cache=False,
            )
            latent_goal_out = outputs2[2]
            latent_goal_v = self.latent_goal_out_proj(
                latent_goal_out[:, -1, :].to(dtype=torch.float32)
            )
            per_sample_fs = F.mse_loss(
                latent_goal_u_t, latent_goal_v, reduction="none"
            ).mean(dim=-1)
            if chunk_end_pad_mask is not None:
                valid = (~chunk_end_pad_mask).to(dtype=per_sample_fs.dtype)
                denom = valid.sum().clamp_min(1.0)
                latent_goal_loss = (per_sample_fs * valid).sum() / denom
            else:
                latent_goal_loss = per_sample_fs.mean()

            # ── Build z_g for the action expert ──
            if self.config.latent_goal_inject_z_g_source == "encoded":
                z_g_for_action = z_g_target
            elif self.config.latent_goal_train_num_steps > 1:
                # K-step iterative denoise — matches the inference loop
                # exactly so the action expert sees the same z_g
                # distribution at train and eval. Runs under no_grad
                # because z_g is detached on the action-expert side; the
                # LGE flow-matching signal still comes from the Pass 2
                # single-t forward above (latent_goal_v).
                with torch.no_grad():
                    z_g_for_action = self._latent_goal_denoise(
                        z_t_anchor, prefix_pad_masks, past_key_values
                    )
            else:
                # One-step closed-form reconstruction.
                # Flow matching: x_t = t·noise + (1-t)·z_g  and  v = noise − z_g
                #   ⇒  z_g_pred = x_t − t · v
                z_g_for_action = (
                    latent_goal_x_t.to(dtype=torch.float32)
                    - latent_goal_t_exp.to(dtype=torch.float32) * latent_goal_v
                )

            z_t_for_action = z_t_anchor
            if self.config.latent_goal_inject_detach:
                z_t_for_action = z_t_for_action.detach()
                z_g_for_action = z_g_for_action.detach()

            zt_emb = self.latent_goal_action_zt_proj(
                z_t_for_action.to(self.latent_goal_action_zt_proj.weight.dtype)
            )
            zg_emb = self.latent_goal_action_zg_proj(
                z_g_for_action.to(self.latent_goal_action_zg_proj.weight.dtype)
            )
            latent_goal_inject = torch.stack([zt_emb, zg_emb], dim=1)  # (B, 2, hidden)

            # ── Pass 3: action expert forward with z_t/z_g prepended ──
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
                x_t, time, suffix_lewm, latent_goal_inject_tokens=latent_goal_inject
            )
            pass3_pad = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            pass3_att = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            pass3_2d = make_att_2d_masks(pass3_pad, pass3_att)
            action_attention_mask = pass3_2d[:, prefix_len:, :]
            pass3_position_ids_full = torch.cumsum(pass3_pad, dim=1) - 1
            action_position_ids = pass3_position_ids_full[:, prefix_len:]

            outputs3, _ = self.vlm_with_expert.forward(
                attention_mask=action_attention_mask,
                position_ids=action_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs, None],
                use_cache=self.config.use_cache,
                fill_kv_cache=False,
            )
            action_suffix_out = outputs3[1][:, -self.config.chunk_size :].to(dtype=torch.float32)
            v_t = self.action_out_proj(action_suffix_out)
            losses = F.mse_loss(u_t, v_t, reduction="none")

        elif self.config.latent_goal_enabled:
            # ─────────────── Phase A: parallel single forward ───────────────
            # Twin-pass — build a 2-token suffix for the Latent Goal Expert
            # (anchor z_t + noisy z_g+time) and run both experts through the
            # shared VLM in one forward.
            if chunk_end_images is None:
                raise ValueError(
                    "latent_goal_enabled=True requires chunk_end_images in forward(); "
                    "the policy wrapper must extract the o_{t+chunk_size} frame from "
                    "the batch when observation_delta_indices=[0, chunk_size]."
                )
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
                x_t, time, suffix_lewm
            )
            z_t_anchor = self._encode_lewm_cls(images)              # (B, 192)
            z_g_target = self._encode_lewm_cls(chunk_end_images)    # (B, 192)

            latent_goal_noise = self.sample_noise(z_g_target.shape, z_g_target.device)
            latent_goal_time = self.sample_time(z_g_target.shape[0], z_g_target.device)
            latent_goal_t_exp = latent_goal_time[:, None]
            latent_goal_x_t = (
                latent_goal_t_exp * latent_goal_noise + (1 - latent_goal_t_exp) * z_g_target
            )
            latent_goal_u_t = latent_goal_noise - z_g_target

            latent_goal_embs, latent_goal_pad_masks, latent_goal_att_masks = (
                self.embed_latent_goal_suffix(z_t_anchor, latent_goal_x_t, latent_goal_time)
            )

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks, latent_goal_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks, latent_goal_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            # Block Latent Goal Expert → suffix attention. The cumulative-mask scheme would
            # otherwise let Latent Goal Expert tokens read from action (and suffix-lewm)
            # tokens since their cumsum sits above the suffix. We want Latent Goal Expert
            # to predict z_g from (prefix, language goal, z_t anchor) only —
            # independent of the action chunk being committed. Action → Latent Goal Expert
            # is already blocked by the cumsum ordering (Latent Goal Expert comes later).
            suffix_len = suffix_pad_masks.shape[1]
            latent_goal_start = prefix_len + suffix_len
            att_2d_masks[:, latent_goal_start:, prefix_len:latent_goal_start] = False
            position_ids = torch.cumsum(pad_masks, dim=1) - 1

            (_, suffix_out, latent_goal_out), _ = self.vlm_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs, latent_goal_embs],
                use_cache=False,
                fill_kv_cache=False,
            )
            action_suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
            v_t = self.action_out_proj(action_suffix_out)
            losses = F.mse_loss(u_t, v_t, reduction="none")

            latent_goal_v = self.latent_goal_out_proj(
                latent_goal_out[:, -1, :].to(dtype=torch.float32)
            )
            per_sample_fs = F.mse_loss(
                latent_goal_u_t, latent_goal_v, reduction="none"
            ).mean(dim=-1)
            if chunk_end_pad_mask is not None:
                valid = (~chunk_end_pad_mask).to(dtype=per_sample_fs.dtype)
                denom = valid.sum().clamp_min(1.0)
                latent_goal_loss = (per_sample_fs * valid).sum() / denom
            else:
                latent_goal_loss = per_sample_fs.mean()

        else:
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
                x_t, time, suffix_lewm
            )
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            (_, suffix_out), _ = self.vlm_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
            )
            suffix_out = suffix_out[:, -self.config.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
            losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses, latent_goal_loss

    def _build_inference_context(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
    ) -> dict:
        """Build the per-step VLM prefix cache + (optionally) LGE state.

        Centralizes the pre-denoising work shared between the standard
        sampling path and Phase B / MPC. Returns a dict with:
          - bsize, device
          - suffix_lewm_tokens (or None)
          - prefix_pad_masks, past_key_values
          - latent_goal_inject_tokens (or None)  — Mode 3 only
          - z_t_cls, z_g_cls (or None each)      — only when LGE active
        """
        bsize = state.shape[0]
        device = state.device

        suffix_lewm_tokens = self.compute_lewm_tokens(images)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        mode3 = (
            self.config.latent_goal_enabled and self.config.latent_goal_inject_to_action
        )
        prefix_inputs = [prefix_embs, None, None] if mode3 else [prefix_embs, None]
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=prefix_inputs,
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        latent_goal_inject_tokens: torch.Tensor | None = None
        z_t_cls: torch.Tensor | None = None
        z_g_cls: torch.Tensor | None = None
        if mode3:
            z_t_cls = self._encode_lewm_cls(images)
            z_g_cls = self._latent_goal_denoise(z_t_cls, prefix_pad_masks, past_key_values)
            zt_emb = self.latent_goal_action_zt_proj(
                z_t_cls.to(self.latent_goal_action_zt_proj.weight.dtype)
            )
            zg_emb = self.latent_goal_action_zg_proj(
                z_g_cls.to(self.latent_goal_action_zg_proj.weight.dtype)
            )
            latent_goal_inject_tokens = torch.stack([zt_emb, zg_emb], dim=1)
        elif self.config.mpc_enabled:
            # MPC without Mode 3 still needs both anchors for predictor
            # rollout + scoring. Compute them here on the same cache.
            z_t_cls = self._encode_lewm_cls(images)
            z_g_cls = self._latent_goal_denoise(z_t_cls, prefix_pad_masks, past_key_values)

        return {
            "bsize": bsize,
            "device": device,
            "suffix_lewm_tokens": suffix_lewm_tokens,
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
            "latent_goal_inject_tokens": latent_goal_inject_tokens,
            "z_t_cls": z_t_cls,
            "z_g_cls": z_g_cls,
        }

    def _denoise_action_chunk(
        self,
        ctx: dict,
        noise: Tensor,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Standard num_steps flow-matching denoising loop.

        Inputs come from ``_build_inference_context``. Output shape:
        (B, chunk_size, max_action_dim) — padded action space.
        """
        device = ctx["device"]
        bsize = ctx["bsize"]
        prefix_pad_masks = ctx["prefix_pad_masks"]
        past_key_values = ctx["past_key_values"]
        suffix_lewm_tokens = ctx["suffix_lewm_tokens"]
        latent_goal_inject_tokens = ctx["latent_goal_inject_tokens"]

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                    lewm_tokens=suffix_lewm_tokens,
                    latent_goal_inject_tokens=latent_goal_inject_tokens,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors).

        When ``mpc_enabled`` is set, this dispatches to MPC: build the
        anchor chunk via the standard denoising loop, then refine it
        against the LGE goal via le-wm predictor rollouts. MPC is
        eval-only by construction — it lives behind ``sample_actions``,
        which is invoked from the policy's ``@torch.no_grad()``
        ``select_action`` / ``predict_action_chunk`` only.
        """
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        ctx = self._build_inference_context(
            images, img_masks, lang_tokens, lang_masks, state
        )

        if self.config.mpc_enabled:
            return self._mpc_sample_actions(ctx, noise, **kwargs)

        return self._denoise_action_chunk(ctx, noise, **kwargs)

    # ── Phase B / MPC inference path ───────────────────────────────────

    def _mpc_sample_actions(
        self,
        ctx: dict,
        noise: Tensor,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Anchor-based MPC. Returns (B, chunk_size, max_action_dim).

        Steps:
          1. Run the policy's standard denoising once to get the clean
             anchor chunk ``a*``.
          2. Compute (z_t, z_g) in the configured scoring space.
          3. Sample N candidate action chunks per the configured scheme,
             roll each through the le-wm predictor, score against z_g.
          4. Return the argmin candidate's chunk, re-padded to max
             action dim.

        The anchor is always candidate 0 in ``anchor_perturb`` so MPC
        cannot make the policy strictly worse — if scoring is
        uncalibrated, the anchor still wins.
        """
        # (1) Anchor pass at original batch size.
        anchor_padded = self._denoise_action_chunk(ctx, noise, **kwargs)
        action_dim = self.config.action_feature.shape[0]
        anchor = anchor_padded[..., :action_dim]  # (B, T, A_raw)

        # (2) Scoring-space latents.
        z_t_emb, z_g_emb = self._build_mpc_targets(ctx)

        # (3) Pick candidates per scheme.
        if self.config.mpc_scheme == "cem":
            best_raw = self._cem_search(z_t_emb, z_g_emb, anchor)
        else:
            best_raw = self._anchor_perturb_search(z_t_emb, z_g_emb, anchor)

        # (4) Re-pad to max_action_dim. The trailing pad slots stay zero
        #     to match what the action expert would emit there.
        out = anchor_padded.clone()
        out[..., :action_dim] = best_raw
        return out

    def _build_mpc_targets(self, ctx: dict) -> tuple[Tensor, Tensor]:
        """Map z_t / z_g into the configured scoring space.

        ``post_proj`` applies le-wm's MLP projector to both — same space
        the predictor was supervised against. ``cls`` keeps both in raw
        CLS space (mismatched against predictor output, but matches
        LGE's training target).
        """
        z_t_cls = ctx["z_t_cls"]
        z_g_cls = ctx["z_g_cls"]
        assert z_t_cls is not None and z_g_cls is not None, (
            "_build_mpc_targets called without LGE state — should be unreachable"
        )
        if self.config.mpc_score_space == "post_proj":
            z_t_emb = self.lewm_world.project(z_t_cls)
            z_g_emb = self.lewm_world.project(z_g_cls)
        else:
            z_t_emb, z_g_emb = z_t_cls, z_g_cls
        return z_t_emb.to(torch.float32), z_g_emb.to(torch.float32)

    def _anchor_perturb_search(
        self,
        z_t_emb: Tensor,
        z_g_emb: Tensor,
        anchor: Tensor,
    ) -> Tensor:
        """Scheme A. Returns best candidate per batch element: (B, T, A_raw)."""
        B, T, A = anchor.shape
        N = self.config.mpc_num_candidates
        sigma = self.config.mpc_noise_scale
        device = anchor.device

        # Candidate 0 = anchor; rest = anchor + sigma * eps.
        eps = torch.randn(B, N - 1, T, A, device=device, dtype=anchor.dtype)
        candidates = torch.cat(
            [anchor.unsqueeze(1), anchor.unsqueeze(1) + sigma * eps],
            dim=1,
        )  # (B, N, T, A)

        cost = self._lewm_rollout_score(z_t_emb, z_g_emb, candidates)  # (B, N)
        best = cost.argmin(dim=1)  # (B,)
        return candidates[torch.arange(B, device=device), best]

    def _cem_search(
        self,
        z_t_emb: Tensor,
        z_g_emb: Tensor,
        anchor: Tensor,
    ) -> Tensor:
        """Scheme B. CEM over Gaussian perturbations centered on the anchor."""
        B, T, A = anchor.shape
        N = self.config.mpc_num_candidates
        sigma_init = self.config.mpc_noise_scale
        blend = self.config.mpc_cem_anchor_blend
        topk = self.config.mpc_cem_topk
        num_iter = self.config.mpc_cem_num_iter
        device = anchor.device
        dtype = anchor.dtype
        batch_idx = torch.arange(B, device=device)

        mu = anchor.clone()  # (B, T, A)
        sigma = torch.full_like(anchor, sigma_init)

        # Track the best candidate ever seen, in case later CEM iters
        # drift to a worse region.
        anchor_cost = self._lewm_rollout_score(z_t_emb, z_g_emb, anchor.unsqueeze(1)).squeeze(1)
        best_cost = anchor_cost
        best_actions = anchor.clone()

        for _ in range(num_iter):
            eps = torch.randn(B, N, T, A, device=device, dtype=dtype)
            candidates = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps  # (B, N, T, A)
            cost = self._lewm_rollout_score(z_t_emb, z_g_emb, candidates)  # (B, N)

            elite_idx = cost.topk(topk, dim=1, largest=False).indices  # (B, K)
            elite = candidates[batch_idx[:, None], elite_idx]  # (B, K, T, A)
            mu = elite.mean(dim=1)
            sigma_new = elite.std(dim=1)
            sigma = sigma_new * (1 - blend) + sigma_init * blend

            iter_best = cost.argmin(dim=1)
            iter_cost = cost[batch_idx, iter_best]
            iter_actions = candidates[batch_idx, iter_best]
            improved = iter_cost < best_cost
            best_cost = torch.where(improved, iter_cost, best_cost)
            improved_mask = improved.view(B, 1, 1).expand_as(best_actions)
            best_actions = torch.where(improved_mask, iter_actions, best_actions)

        return best_actions

    def _lewm_rollout_score(
        self,
        z_t_emb: Tensor,
        z_g_emb: Tensor,
        candidates: Tensor,
    ) -> Tensor:
        """Roll candidate action chunks through le-wm and score vs z_g.

        Args:
            z_t_emb: (B, 192) current latent in scoring space.
            z_g_emb: (B, 192) goal latent in scoring space.
            candidates: (B, N, T, A_raw) candidate action chunks.

        Returns:
            cost: (B, N) sum-of-squares distance between the final
            predicted latent and z_g.

        History fabrication: le-wm trained with history_size=3; we only
        have one real frame (z_t). We pad to HS=3 by repeating z_t and
        prefix HS-1 zero actions, so the first prediction effectively
        treats the past as static. Each iter then takes the last HS
        (emb, action) pairs and predicts the next frame. Only the final
        emb (z_{t+T}) feeds the cost.
        """
        assert self.lewm_world is not None, "MPC enabled without lewm_world loaded"
        B, N, T, A = candidates.shape
        HS = self.lewm_world.history_size
        device = candidates.device

        flat = candidates.reshape(B * N, T, A)
        z_t_flat = z_t_emb.repeat_interleave(N, dim=0)  # (B*N, 192)
        z_g_flat = z_g_emb.repeat_interleave(N, dim=0)  # (B*N, 192)

        hist_emb = z_t_flat.unsqueeze(1).expand(-1, HS, -1).contiguous()  # (B*N, HS, 192)
        # Zero actions for the fabricated past (the HS-1 "before-now" slots).
        # Each predict call sees the last HS (emb, action) pairs and
        # predicts the next emb at position -1.
        hist_act = torch.zeros(B * N, HS - 1, A, device=device, dtype=flat.dtype)

        for t in range(T):
            hist_act = torch.cat([hist_act, flat[:, t : t + 1, :]], dim=1)
            emb_win = hist_emb[:, -HS:]
            act_win = hist_act[:, -HS:]
            act_emb = self.lewm_world.encode_actions(act_win)
            pred = self.lewm_world.predict_step(emb_win, act_emb)  # (B*N, HS, 192)
            hist_emb = torch.cat([hist_emb, pred[:, -1:, :].to(hist_emb.dtype)], dim=1)

        z_final = hist_emb[:, -1].to(torch.float32)  # (B*N, 192)
        cost = ((z_final - z_g_flat) ** 2).sum(dim=-1)  # (B*N,)
        return cost.view(B, N)

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        lewm_tokens: torch.Tensor | None = None,
        latent_goal_inject_tokens: torch.Tensor | None = None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep.

        ``latent_goal_inject_tokens`` (Mode 3 only) is a (B, 2, hidden)
        tensor of pre-projected [z_t, z_g] tokens; static across denoising
        steps. When provided, the wrapper is invoked through the n-stream
        path so the latent_goal_expert column is silenced (None) while the
        action expert reads the cached VLM K/V.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            x_t, timestep, lewm_tokens, latent_goal_inject_tokens=latent_goal_inject_tokens
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        action_inputs_embeds = (
            [None, suffix_embs, None]
            if (self.config.latent_goal_enabled and self.config.latent_goal_inject_to_action)
            else [None, suffix_embs]
        )
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=action_inputs_embeds,
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

    def _latent_goal_denoise(
        self,
        z_t_anchor: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
    ) -> torch.Tensor:
        """K-step flow-matching denoising of the LGE goal latent at inference.

        Produces a clean ``z_g`` (B, 192) by initializing from Gaussian
        noise at t=1 and integrating the LGE velocity field down to t=0
        with ``num_steps=config.latent_goal_num_steps``. Reuses the
        prefix VLM K/V cache that was built before action denoising
        starts, so the only marginal cost vs Phase A is the LGE expert
        forwards (one per denoising step).
        """
        bsize = z_t_anchor.shape[0]
        device = z_t_anchor.device
        latent_dim = z_t_anchor.shape[-1]
        z = self.sample_noise((bsize, latent_dim), device)
        num_steps = self.config.latent_goal_num_steps
        dt = -1.0 / num_steps
        for step in range(num_steps):
            t_scalar = 1.0 + step * dt
            t = torch.tensor(t_scalar, dtype=torch.float32, device=device).expand(bsize)
            v = self._latent_goal_denoise_step(z_t_anchor, z, t, prefix_pad_masks, past_key_values)
            z = z + dt * v
        return z

    def _latent_goal_denoise_step(
        self,
        z_t_anchor: torch.Tensor,
        noisy_z: torch.Tensor,
        timestep: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
    ) -> torch.Tensor:
        """One LGE denoising step. Mirrors the action ``denoise_step`` plumbing
        for the LGE suffix; emits the velocity at the denoising token
        position."""
        latent_goal_embs, latent_goal_pad_masks, latent_goal_att_masks = (
            self.embed_latent_goal_suffix(z_t_anchor, noisy_z, timestep)
        )
        suffix_len = latent_goal_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d = make_att_2d_masks(latent_goal_pad_masks, latent_goal_att_masks)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(latent_goal_pad_masks, dim=1) - 1

        outputs, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, latent_goal_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        latent_goal_out = outputs[2]
        return self.latent_goal_out_proj(latent_goal_out[:, -1, :].to(dtype=torch.float32))
