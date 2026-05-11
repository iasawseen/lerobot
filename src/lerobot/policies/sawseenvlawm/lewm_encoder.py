# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Wrapper for the le-wm (LeWorldModel JEPA) ViT-Tiny encoder, used as an
extra visual stream feeding the SawSeenVLA action expert.

The le-wm encoder is a HuggingFace `ViTModel` (Tiny: hidden=192, patch=14,
image=224, 12 layers, 3 heads). Trained as part of a JEPA world model on
Libero, and saved (via `torch.save(world_model, …)`) as a pickled `JEPA`
module. We pull the encoder + projector out of that pickle and ignore the
predictor / action_encoder / pred_proj sub-modules.

The forward path:
  1. Accept SawSeenVLA-formatted images: float tensor in [-1, 1], (B, 3, H, W)
  2. Map back to [0, 1], then apply ImageNet mean/std (as during le-wm
     training; see le-wm/utils.py:get_img_preprocessor).
  3. Bilinearly resize to the ViT's training resolution (224×224) if needed.
  4. Run the ViT with `interpolate_pos_encoding=True` so any minor mismatch
     in patch grid still works.
  5. Slice the requested number of tokens from `last_hidden_state`. We take
     ``[:, :num_tokens]`` which keeps CLS + (num_tokens - 1) patches.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import require_package

# ImageNet stats — same values stable_pretraining uses (ImageNet1K means/stds).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_vit_tiny(image_size: int = 224, patch_size: int = 14):
    """Build an empty HF ViTModel matching the le-wm Tiny configuration.

    Mirrors `stable_pretraining.backbone.utils.vit_hf("tiny", patch_size=14,
    image_size=224, pretrained=False, use_mask_token=False)`.
    """
    require_package("transformers", extra="smolvla")
    from transformers import ViTConfig, ViTModel

    config = ViTConfig(
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=192 * 4,
        hidden_act="gelu",
        image_size=image_size,
        patch_size=patch_size,
        num_channels=3,
        qkv_bias=True,
    )
    return ViTModel(config, add_pooling_layer=False, use_mask_token=False)


class LeWMVisionEncoder(nn.Module):
    """ViT-Tiny encoder lifted from a le-wm JEPA checkpoint.

    The input is bilinearly resized to ``(image_height, image_width)`` and
    forwarded through the ViT with ``interpolate_pos_encoding=True`` so the
    HF model accepts non-square / non-training shapes. le-wm trained on
    horizontally concatenated camera pairs (e.g. libero's 256×512), so the
    natural shape for downstream parity is *rectangular* — set width to
    ``num_cameras × height``.

    Args:
        num_tokens: how many tokens to expose to downstream layers, sliced
            from ``last_hidden_state[:, :num_tokens]``. Index 0 is the CLS
            token, indices 1.. are patch tokens in row-major order.
        image_height, image_width: target ViT input resolution.
        patch_size: ViT patch size.
        freeze: whether to freeze all encoder parameters.
    """

    output_dim: int = 192  # ViT-Tiny hidden size

    def __init__(
        self,
        num_tokens: int = 192,
        image_height: int = 224,
        image_width: int = 448,
        patch_size: int = 14,
        freeze: bool = True,
    ):
        super().__init__()
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
        if image_height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError(
                f"image dims ({image_height}, {image_width}) must be divisible by patch_size {patch_size}"
            )
        max_tokens = (image_height // patch_size) * (image_width // patch_size) + 1
        if num_tokens > max_tokens:
            raise ValueError(
                f"num_tokens={num_tokens} exceeds the ViT-Tiny grid "
                f"({max_tokens}: 1 CLS + {max_tokens - 1} patches at "
                f"{image_height}×{image_width}/{patch_size})."
            )

        self.num_tokens = num_tokens
        self.image_height = image_height
        self.image_width = image_width

        # ViT's `image_size` must match the checkpoint's pos_emb shape (le-wm
        # used image_size=224 → 16x16+1=257 tokens of pos_emb in the .ckpt).
        # `interpolate_pos_encoding=True` at forward time re-fits the
        # positional embedding to whatever (image_height, image_width) we feed.
        self.vit = _build_vit_tiny(image_size=224, patch_size=patch_size)

        # LeWM's projector: MLP(192 → 2048 → 192) with BatchNorm — maps raw
        # ViT CLS into the JEPA prediction space (where `pred_proj` also
        # lands). Populated by ``from_lewm_checkpoint``; ``encode_cls`` is
        # the canonical "scalar latent for an image" entrypoint and applies
        # it. ``forward`` (returning multi-token slices for the lewm
        # side-channel) keeps raw ViT hidden states — projector was
        # supervised on CLS only, so applying it to patch tokens is OOD.
        self.projector: nn.Module | None = None

        # ImageNet normalization buffers (broadcastable to (B, 3, H, W)).
        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

        self.freeze = freeze
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> "LeWMVisionEncoder":
        """Override to keep frozen sub-modules in eval mode.

        The projector is an MLP with BatchNorm1d (from le-wm training).
        When ``freeze=True`` we never want it to recompute running stats
        on SawSeenVLAWM batches — those would corrupt the frozen feature
        distribution. ``requires_grad=False`` alone doesn't gate
        BatchNorm; the explicit ``.eval()`` propagation here does.
        """
        super().train(mode)
        if self.freeze:
            self.vit.eval()
            if self.projector is not None:
                self.projector.eval()
        return self

    @classmethod
    def from_lewm_checkpoint(
        cls,
        ckpt_path: str | Path,
        num_tokens: int = 192,
        image_height: int = 224,
        image_width: int = 448,
        patch_size: int = 14,
        freeze: bool = True,
    ) -> "LeWMVisionEncoder":
        """Load encoder weights from a le-wm `<name>_object.ckpt`.

        These checkpoints are produced by le-wm/utils.py:_dump_model via
        `torch.save(world_model, ...)` on a `JEPA` module, so we load with
        ``weights_only=False``.
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"le-wm checkpoint not found: {ckpt_path}")

        # The JEPA module imports from le-wm's `module` and `jepa` packages.
        # We don't need to actually instantiate it — just access .encoder's
        # state_dict. But torch.load(weights_only=False) will run __reduce__
        # which requires those classes to be importable. Allow that by
        # falling back to a pure state_dict load if the pickle import fails.
        encoder = cls(
            num_tokens=num_tokens,
            image_height=image_height,
            image_width=image_width,
            patch_size=patch_size,
            freeze=freeze,
        )
        try:
            obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to unpickle {ckpt_path}. The le-wm `JEPA` and "
                f"`module` python files must be importable. Add le-wm to "
                f"PYTHONPATH or vendor the classes. Original error: {e}"
            ) from e

        # The pickled object is a JEPA module; .encoder is the HF ViT and
        # .projector is the post-encoder MLP (le-wm/module.py:MLP).
        for required in ("encoder", "projector"):
            if not hasattr(obj, required):
                raise RuntimeError(
                    f"Unexpected checkpoint structure in {ckpt_path}: missing "
                    f".{required}. Got type {type(obj).__name__}."
                )
        encoder_state = obj.encoder.state_dict()
        missing, unexpected = encoder.vit.load_state_dict(encoder_state, strict=False)
        if missing or unexpected:
            # ViTModel adds nothing extra over what stable_pretraining's
            # vit_hf wraps, so missing/unexpected should be empty in practice.
            # We log if they aren't, but don't fail — the encoder still works.
            import logging

            logging.getLogger(__name__).warning(
                "le-wm encoder load: missing=%s unexpected=%s",
                list(missing)[:5],
                list(unexpected)[:5],
            )

        # Attach the projector as-is — full nn.Module with state already
        # loaded from the pickle (le-wm/module.py:MLP).
        encoder.projector = obj.projector

        if freeze:
            for p in encoder.parameters():
                p.requires_grad = False
            # Force eval mode on BatchNorm-bearing sub-modules. The
            # ``train()`` override below keeps them eval-locked, but we
            # also set it explicitly at load so the first forward (before
            # any ``.train()`` call) is correct.
            encoder.vit.eval()
            encoder.projector.eval()
        return encoder

    def _normalize(self, img: Tensor) -> Tensor:
        """Convert from SawSeenVLA's [-1, 1] range back to ImageNet normalized."""
        # img: (B, 3, H, W) in [-1, 1]
        x = (img + 1.0) * 0.5  # → [0, 1]
        x = (x - self._mean.to(x.dtype)) / self._std.to(x.dtype)
        return x

    @torch.no_grad()
    def _maybe_resize(self, img: Tensor) -> Tensor:
        if img.shape[-2] != self.image_height or img.shape[-1] != self.image_width:
            img = F.interpolate(
                img,
                size=(self.image_height, self.image_width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return img

    def forward(self, img: Tensor) -> Tensor:
        """Encode an (already-concatenated) image into ViT tokens.

        Args:
            img: (B, 3, H, W) float, in [-1, 1] (SawSeenVLA's prepare_images
                output). For multi-camera setups, callers should
                concatenate the per-camera images horizontally first so the
                ViT sees the same layout it was trained on.
        Returns:
            tokens: (B, num_tokens, 192).
        """
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, H, W); got {tuple(img.shape)}")

        x = self._maybe_resize(img)
        x = self._normalize(x)
        # Cast back to whatever dtype the ViT params live in (e.g. fp32 even
        # under bf16 autocast — the encoder is frozen so we don't need bf16).
        param = next(self.vit.parameters())
        x = x.to(dtype=param.dtype)

        # Frozen path: no_grad on the ViT itself when freeze=True. We let
        # autocast still wrap it to be safe under accelerate/fp16.
        if self.freeze:
            with torch.no_grad():
                out = self.vit(pixel_values=x, interpolate_pos_encoding=True)
        else:
            out = self.vit(pixel_values=x, interpolate_pos_encoding=True)

        hidden = out.last_hidden_state  # (B, 1+P*P, 192)
        return hidden[:, : self.num_tokens]

    def encode_cls(self, img: Tensor) -> Tensor:
        """Encode an image to a single 192-d latent in the LeWM JEPA
        prediction space (post-projector).

        This is the canonical "what does LeWM think this image looks
        like, as a single vector" call. Output matches the space the
        LeWM predictor was supervised against, so it is directly
        comparable to ``LeWMWorldModel.predict_step`` outputs and to
        anything LGE produces (LGE is trained against this same space).

        Args:
            img: (B, 3, H, W) float in [-1, 1]; same camera-concatenated
                layout as ``forward``.
        Returns:
            emb: (B, 192) post-projector embedding.
        """
        assert self.projector is not None, (
            "LeWMVisionEncoder.projector not loaded — call from_lewm_checkpoint"
        )
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, H, W); got {tuple(img.shape)}")
        x = self._maybe_resize(img)
        x = self._normalize(x)
        vit_param = next(self.vit.parameters())
        x = x.to(dtype=vit_param.dtype)
        proj_param = next(self.projector.parameters())

        if self.freeze:
            with torch.no_grad():
                out = self.vit(pixel_values=x, interpolate_pos_encoding=True)
                cls = out.last_hidden_state[:, 0, :]  # (B, 192)
                emb = self.projector(cls.to(proj_param.dtype))
        else:
            out = self.vit(pixel_values=x, interpolate_pos_encoding=True)
            cls = out.last_hidden_state[:, 0, :]
            emb = self.projector(cls.to(proj_param.dtype))
        return emb


class LeWMWorldModel(nn.Module):
    """Full le-wm JEPA module lifted from a `<name>_object.ckpt` pickle.

    Wraps the same `LeWMVisionEncoder` (for image preprocessing + ViT) and
    additionally exposes the projector, action_encoder, predictor, and
    pred_proj sub-modules. Used by Phase B / MPC inference to roll
    candidate action chunks forward in latent space.

    All sub-modules are frozen by default — this is an inference-only
    consumer; the SawSeenVLAWM training loss path never touches it.
    """

    output_dim: int = 192  # ViT-Tiny hidden / post-projector emb dim
    history_size: int = 3  # le-wm `wm.history_size` default

    def __init__(
        self,
        num_tokens: int = 192,
        image_height: int = 224,
        image_width: int = 448,
        patch_size: int = 14,
        freeze: bool = True,
    ):
        super().__init__()
        # Reuses the vision wrapper's normalize/resize/cast path and ViT.
        # The projector lives on ``_vision`` so ``encode_cls`` works on
        # both this wrapper and the standalone encoder consistently.
        self._vision = LeWMVisionEncoder(
            num_tokens=num_tokens,
            image_height=image_height,
            image_width=image_width,
            patch_size=patch_size,
            freeze=freeze,
        )
        # Populated by ``from_lewm_checkpoint`` — typed as Module so torch
        # registers them; None until loaded.
        self.action_encoder: nn.Module | None = None
        self.predictor: nn.Module | None = None
        self.pred_proj: nn.Module | None = None
        self.freeze = freeze

    @property
    def encoder(self) -> LeWMVisionEncoder:
        return self._vision

    @property
    def projector(self) -> nn.Module | None:
        """Convenience pointer to ``self._vision.projector`` (the single
        loaded copy). Predates the refactor where the world model held a
        separate slot; kept for backwards compatibility."""
        return self._vision.projector

    def train(self, mode: bool = True) -> "LeWMWorldModel":
        """Override to keep frozen sub-modules in eval mode.

        ``_vision.train()`` handles the ViT + projector. We additionally
        force the predictor side back to eval — ``pred_proj`` has
        BatchNorm1d (le-wm/module.py:MLP with norm_fn=BatchNorm1d) and
        must not update running stats on SawSeenVLAWM batches. The
        predictor + action_encoder are LayerNorm-only / norm-free, so
        the call is technically a no-op for them but harmless.
        """
        super().train(mode)
        if self.freeze:
            # _vision.train() already eval-locks vit + projector; calling
            # it explicitly here is redundant (super().train propagates)
            # but defensive.
            self._vision.eval()
            if self.predictor is not None:
                self.predictor.eval()
            if self.pred_proj is not None:
                self.pred_proj.eval()
            if self.action_encoder is not None:
                self.action_encoder.eval()
        return self

    @classmethod
    def from_lewm_checkpoint(
        cls,
        ckpt_path: str | Path,
        num_tokens: int = 192,
        image_height: int = 224,
        image_width: int = 448,
        patch_size: int = 14,
        freeze: bool = True,
    ) -> "LeWMWorldModel":
        """Load encoder + projector + action_encoder + predictor + pred_proj
        from a le-wm `<name>_object.ckpt`.

        Same pickle as `LeWMVisionEncoder.from_lewm_checkpoint`, but
        retains all five sub-modules instead of dropping the predictor
        side. The JEPA / module python files must be importable to unpickle.
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"le-wm checkpoint not found: {ckpt_path}")

        model = cls(
            num_tokens=num_tokens,
            image_height=image_height,
            image_width=image_width,
            patch_size=patch_size,
            freeze=freeze,
        )
        try:
            obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to unpickle {ckpt_path}. The le-wm `JEPA` and "
                f"`module` python files must be importable. Add le-wm to "
                f"PYTHONPATH or vendor the classes. Original error: {e}"
            ) from e

        for required in ("encoder", "projector", "action_encoder", "predictor", "pred_proj"):
            if not hasattr(obj, required):
                raise RuntimeError(
                    f"Unexpected checkpoint structure in {ckpt_path}: missing "
                    f".{required}. Got type {type(obj).__name__}."
                )

        encoder_state = obj.encoder.state_dict()
        missing, unexpected = model._vision.vit.load_state_dict(encoder_state, strict=False)
        if missing or unexpected:
            import logging

            logging.getLogger(__name__).warning(
                "le-wm encoder load: missing=%s unexpected=%s",
                list(missing)[:5],
                list(unexpected)[:5],
            )

        # Attach the rest as-is. They are full nn.Module instances with
        # their state loaded — assignment registers them on the parent.
        # Projector goes onto _vision (single canonical holder) so the
        # encoder wrapper's encode_cls works too.
        model._vision.projector = obj.projector
        model.action_encoder = obj.action_encoder
        model.predictor = obj.predictor
        model.pred_proj = obj.pred_proj

        if freeze:
            for p in model.parameters():
                p.requires_grad = False
            # Force eval mode on BatchNorm-bearing sub-modules. `train()`
            # below keeps them locked; this handles the load-time path
            # before any explicit `.train()` call.
            model._vision.vit.eval()
            model._vision.projector.eval()
            model.pred_proj.eval()
            model.predictor.eval()
            model.action_encoder.eval()
        return model

    def encode_cls(self, images: list[Tensor]) -> Tensor:
        """Encode camera-concatenated images and return the post-projector
        latent (B, 192).

        Delegates to ``self._vision.encode_cls`` which applies the LeWM
        projector — output is in the JEPA prediction space, directly
        comparable to ``predict_step`` outputs.
        """
        if len(images) == 1:
            stacked = images[0]
        else:
            stacked = torch.cat(images, dim=-1)
        return self._vision.encode_cls(stacked)

    @torch.no_grad()
    def encode_actions(self, actions: Tensor) -> Tensor:
        """Run le-wm's action_encoder. ``actions``: (B, T, A_raw); returns (B, T, 192)."""
        assert self.action_encoder is not None, "LeWMWorldModel not loaded — call from_lewm_checkpoint"
        param = next(self.action_encoder.parameters())
        return self.action_encoder(actions.to(param.dtype))

    @torch.no_grad()
    def predict_step(self, emb: Tensor, act_emb: Tensor) -> Tensor:
        """One predictor call: (emb, act_emb) → predicted emb sequence.

        ``emb``: (B, T, 192), ``act_emb``: (B, T, 192). Output (B, T, 192)
        where position i predicts emb at time i+1 (matches le-wm training:
        `ctx_emb = emb[:, :ctx_len]`, `tgt_emb = emb[:, n_preds:]`).
        Callers should take ``[:, -1:, :]`` as the "next frame" prediction
        after the last action in the window.
        """
        assert self.predictor is not None and self.pred_proj is not None, (
            "LeWMWorldModel not loaded — call from_lewm_checkpoint"
        )
        param = next(self.predictor.parameters())
        emb = emb.to(param.dtype)
        act_emb = act_emb.to(param.dtype)
        preds = self.predictor(emb, act_emb)  # (B, T, hidden=192)
        B, T, D = preds.shape
        preds = self.pred_proj(preds.reshape(B * T, D))
        return preds.reshape(B, T, -1)
