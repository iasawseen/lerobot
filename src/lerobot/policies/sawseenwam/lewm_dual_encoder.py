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

"""Wrapper for the **new** le-wm JEPA used by SawSeenWAM.

Differences from ``sawseenvlawm.lewm_encoder.LeWMVisionEncoder /
LeWMWorldModel``:

* The pickled JEPA has ``encoders: nn.ModuleList`` (one ViT per camera)
  and ``pixel_keys = ("pixels", "pixels_wrist")`` for LIBERO — not a
  single ViT over a camera-concat 224×448 image. Camera streams stay
  *separate* through the encoder; only the projector fuses them
  (single-token: concat CLS across cams → MLP; multi-token: per-cam
  query reducer → per-token MLP).

* The action_encoder is variable-stride-aware — accepts per-slot ``k``
  (shape ``(B, T)`` tensor) and a chunked ``(B, T, k_max, A)`` action
  tensor. Slot ``i``'s ``k[i]`` says "this slot's action describes a
  k-step jump forward". Used by the multi-offset cost methods.

* The predictor may be ``ARPredictor`` (AdaLN, legacy compatible) or
  ``SpatioTemporalARPredictor`` (Plan-B multi-token). We don't care
  which — we just call ``self.jepa.predict(emb, act_emb, action_tokens)``.

The JEPA pickle is loaded as-is via ``torch.load(weights_only=False)``;
``le-wm/`` must be importable for the unpickle to resolve class paths
(``module.JEPA``, ``module.PadFlattenActionEncoder``, etc).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class LeWMDualWorldModel(nn.Module):
    """Thin wrapper around the new le-wm JEPA module.

    Responsibilities:
      * Load the pickled JEPA and expose its sub-modules (encoders,
        projector, action_encoder, predictor, pred_proj) for inspection.
      * Image preprocessing: SawSeenVLA images arrive in ``[-1, 1]``;
        the new le-wm expects ImageNet-normalized 224×224 per camera.
        ``encode_states`` handles that.
      * Encode/predict helpers (``encode_states``, ``encode_actions``,
        ``predict_step``) so the policy doesn't have to reach into
        ``self.jepa`` directly.

    Sub-modules are frozen by default (this is an inference / side-channel
    consumer; the SawSeenWAM training-loss path never updates them).
    """

    # Per-camera ViT input. The new le-wm trained each ViT independently
    # at 224×224.
    image_height: int = 224
    image_width: int = 224
    history_size: int = 3
    output_dim: int = 192  # JEPA embed_dim (post-projector)

    def __init__(
        self,
        pixel_keys: tuple[str, ...] = ("pixels", "pixels_wrist"),
        image_height: int = 224,
        image_width: int = 224,
        freeze: bool = True,
    ):
        super().__init__()
        if len(pixel_keys) < 1:
            raise ValueError("pixel_keys must be non-empty")
        self.pixel_keys = tuple(pixel_keys)
        self.image_height = image_height
        self.image_width = image_width
        self.freeze = freeze

        # Populated by ``from_lewm_checkpoint``. The JEPA module owns its
        # encoders/projector/action_encoder/predictor/pred_proj.
        self.jepa: nn.Module | None = None

        # ImageNet stats — broadcastable to (B, 3, H, W).
        self.register_buffer(
            "_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    @property
    def multi_token(self) -> bool:
        """True iff the loaded JEPA has Plan-B query reducers."""
        if self.jepa is None:
            return False
        return getattr(self.jepa, "multi_token", False)

    @property
    def num_cameras(self) -> int:
        """Number of separate encoders the loaded JEPA has."""
        if self.jepa is None or not hasattr(self.jepa, "encoders"):
            return 0
        return len(self.jepa.encoders)

    @classmethod
    def from_lewm_checkpoint(
        cls,
        ckpt_path: str | Path,
        pixel_keys: tuple[str, ...] = ("pixels", "pixels_wrist"),
        image_height: int = 224,
        image_width: int = 224,
        freeze: bool = True,
    ) -> "LeWMDualWorldModel":
        """Load a pickled JEPA module from ``<name>_object.ckpt``.

        Validates that the pickle is the new dual-encoder format. Will
        raise if loaded from an old single-encoder pickle (the latter
        belongs in ``sawseenvlawm.lewm_encoder``).
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"le-wm checkpoint not found: {ckpt_path}")

        wrapper = cls(
            pixel_keys=pixel_keys,
            image_height=image_height,
            image_width=image_width,
            freeze=freeze,
        )
        try:
            jepa = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to unpickle {ckpt_path}. The new le-wm `module` and "
                f"`jepa` python files must be importable (add le-wm root to "
                f"PYTHONPATH). Original error: {e}"
            ) from e

        # Validate structural shape — new lewm only.
        for required in ("encoders", "projector", "action_encoder", "predictor", "pred_proj"):
            if not hasattr(jepa, required):
                raise RuntimeError(
                    f"Unexpected checkpoint structure in {ckpt_path}: missing "
                    f".{required}. This wrapper requires the new dual-encoder "
                    f"le-wm format; for the old single-encoder format use "
                    f"sawseenvlawm.lewm_encoder.LeWMWorldModel instead."
                )
        if not isinstance(jepa.encoders, nn.ModuleList):
            raise RuntimeError(
                f".encoders must be nn.ModuleList; got {type(jepa.encoders).__name__}. "
                f"This looks like the old single-encoder format."
            )
        if len(jepa.encoders) != len(pixel_keys):
            raise RuntimeError(
                f"checkpoint has {len(jepa.encoders)} encoders but pixel_keys "
                f"has {len(pixel_keys)} entries ({pixel_keys}); they must agree."
            )
        # Cross-check the checkpoint's own pixel_keys if present (it should be).
        ckpt_pixel_keys = getattr(jepa, "pixel_keys", None)
        if ckpt_pixel_keys is not None and tuple(ckpt_pixel_keys) != tuple(pixel_keys):
            raise RuntimeError(
                f"pixel_keys mismatch: checkpoint expects {tuple(ckpt_pixel_keys)} but "
                f"got {tuple(pixel_keys)}. The CLAUDE.md for the new lewm notes this "
                f"is a load-bearing convention — encoder i is bound to pixel_keys[i]."
            )

        wrapper.jepa = jepa

        if freeze:
            for p in wrapper.parameters():
                p.requires_grad = False
            wrapper.jepa.eval()
            # Force BN-bearing sub-modules (projector / pred_proj are
            # MLP+BN1d) into eval so they never update running stats.
            for sub in (jepa.projector, jepa.pred_proj):
                if isinstance(sub, nn.Module):
                    sub.eval()

        return wrapper

    def train(self, mode: bool = True) -> "LeWMDualWorldModel":
        """Override to keep frozen sub-modules in eval mode (BN-aware)."""
        super().train(mode)
        if self.freeze and self.jepa is not None:
            self.jepa.eval()
            for sub in (self.jepa.projector, self.jepa.pred_proj):
                if isinstance(sub, nn.Module):
                    sub.eval()
        return self

    # ── image prep ───────────────────────────────────────────────────

    def _normalize(self, img: Tensor) -> Tensor:
        """SawSeenVLA images are in [-1, 1]; new le-wm expects ImageNet-norm."""
        x = (img + 1.0) * 0.5  # → [0, 1]
        return (x - self._mean.to(x.dtype)) / self._std.to(x.dtype)

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

    def _prep(self, img: Tensor) -> Tensor:
        """SawSeenVLA (B,3,H,W) in [-1,1] → ImageNet-normalized (B,3,224,224)."""
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, H, W); got {tuple(img.shape)}")
        x = self._maybe_resize(img)
        x = self._normalize(x)
        # Cast to whatever dtype the encoder params live in.
        param = next(self.jepa.encoders[0].parameters())
        return x.to(dtype=param.dtype)

    # ── encode / predict helpers ─────────────────────────────────────

    def encode_states(self, images: dict[str, Tensor]) -> Tensor:
        """Encode a *batch of single-state* images via the new JEPA.

        Args:
            images: ``{pixel_key: (B, 3, H, W)}`` for each entry in
                ``self.pixel_keys``. Other keys are tolerated and ignored.

        Returns:
            ``(B, D)`` single-token or ``(B, K, D)`` multi-token — the
            post-projector latent in JEPA prediction space.
        """
        assert self.jepa is not None, "LeWMDualWorldModel not loaded"
        info: dict = {}
        for key in self.pixel_keys:
            if key not in images:
                raise KeyError(f"missing pixel key {key!r} (have: {list(images.keys())})")
            img = self._prep(images[key])  # (B, 3, 224, 224)
            info[key] = img.unsqueeze(1)  # add T axis: (B, 1, 3, 224, 224)

        if self.freeze:
            with torch.no_grad():
                info = self.jepa.encode(info)
        else:
            info = self.jepa.encode(info)
        emb = info["emb"]  # (B, 1, D) or (B, 1, K, D)
        return emb[:, 0]  # drop T

    def encode_actions(self, actions: Tensor, k: Tensor | None) -> tuple[Tensor, Tensor | None]:
        """Run the loaded JEPA's action_encoder with optional per-slot k.

        Args:
            actions: legacy ``(B, T, A)`` or chunked ``(B, T, k_max, A)``.
            k: optional per-slot k tensor of shape ``(B, T)``, or None for
                legacy behavior. For variable-stride encoders, ``k`` is
                required and selects ``k_emb`` per slot.
        Returns:
            ``(act_emb, action_tokens)`` where ``action_tokens`` is None
            for non-hybrid encoders (the common case).
        """
        assert self.jepa is not None, "LeWMDualWorldModel not loaded"
        encoder = self.jepa.action_encoder
        param = next(encoder.parameters())
        actions = actions.to(param.dtype)
        if k is not None:
            out = encoder(actions, k)
        else:
            out = encoder(actions)
        if isinstance(out, dict):
            return out["summary"], out.get("tokens")
        return out, None

    @torch.no_grad()
    def predict_step(
        self,
        emb: Tensor,
        act_emb: Tensor,
        action_tokens: Tensor | None = None,
    ) -> Tensor:
        """One ``self.jepa.predict()`` call, no rollout. Used by both
        single-AR and var-stride paths.

        ``emb``: ``(B, T, D)`` or ``(B, T, K, D)``.
        ``act_emb``: ``(B, T, A_emb)``.
        Returns same shape as ``emb`` — position ``i`` predicts state at
        ``i + num_preds`` (typically ``i + 1``).
        """
        assert self.jepa is not None
        return self.jepa.predict(emb, act_emb, action_tokens=action_tokens)


class LeWMDualVisionEncoder(nn.Module):
    """Thin alias used by the side-channel path that wants per-state
    tokens (not the full predictor).

    Holds a reference to the same loaded JEPA as ``LeWMDualWorldModel``
    and exposes:
      * ``encode_cls(images)`` — (B, D) post-projector latent
      * ``encode_tokens(images)`` — (B, n_cam, D) per-cam CLS tokens for
        the action expert's suffix injection. Each token is the per-cam
        post-projector latent stacked along a "token" axis; this differs
        from the old single-encoder forward which returned a long
        sequence of patch tokens.

    Convenience: building this from an existing ``LeWMDualWorldModel`` so
    we don't double-load the pickle.
    """

    output_dim: int = 192

    def __init__(self, world: LeWMDualWorldModel):
        super().__init__()
        self.world = world

    @property
    def num_cameras(self) -> int:
        return self.world.num_cameras

    @property
    def freeze(self) -> bool:
        return self.world.freeze

    def encode_cls(self, images: dict[str, Tensor]) -> Tensor:
        """Single post-projector latent per batch element.

        Returns ``(B, D)`` single-token or ``(B, K, D)`` multi-token; the
        caller flattens / pools as needed for downstream consumers.
        """
        return self.world.encode_states(images)

    def encode_tokens(self, images: dict[str, Tensor]) -> Tensor:
        """Per-camera post-projector latents stacked along a token axis.

        Returns ``(B, n_cam, D)`` (single-token, one CLS-derived token per
        cam) or ``(B, K, D)`` (multi-token, K = (Q+1) * n_cam, already
        per-token).

        Implementation: for single-token, we call the JEPA encoder
        per-camera and apply the projector per-camera. The legacy single-
        token JEPA encode concatenates across cams *before* the projector,
        which would give a single fused token here; to expose per-cam
        tokens we run the projector on each cam separately. This means
        the side-channel sees cam-disentangled features even when the
        JEPA's own scoring path fuses them.

        TODO: for multi-token (Plan B) we currently fall through to
        ``encode_states`` and trust the JEPA's (B, K, D) output — that
        already exposes per-token granularity.
        """
        assert self.world.jepa is not None, "LeWMDualWorldModel not loaded"
        if self.world.multi_token:
            return self.world.encode_states(images)

        # Single-token, per-camera split.
        jepa = self.world.jepa
        cls_per_cam: list[Tensor] = []
        for cam_idx, key in enumerate(self.world.pixel_keys):
            if key not in images:
                raise KeyError(f"missing pixel key {key!r}")
            x = self.world._prep(images[key])  # (B, 3, 224, 224)
            param = next(jepa.encoders[cam_idx].parameters())
            x = x.to(dtype=param.dtype)
            if self.world.freeze:
                with torch.no_grad():
                    out = jepa.encoders[cam_idx](x, interpolate_pos_encoding=True)
            else:
                out = jepa.encoders[cam_idx](x, interpolate_pos_encoding=True)
            cls = out.last_hidden_state[:, 0]  # (B, D_vit)
            # Per-camera projector: the JEPA's projector input dim is
            # ``n_cam * D_vit`` (legacy single-token concat path), so we
            # cannot apply it to a single-cam CLS directly. Instead, zero-
            # pad the other cams' slots so the projector sees the right
            # input shape with only this cam's information.
            cls_per_cam.append(cls)

        # Build the (B, n_cam * D_vit) input the projector expects.
        # For per-cam tokens, we want a separate projector forward per
        # cam — fill in this cam's slot, zero the others.
        D_vit = cls_per_cam[0].shape[-1]
        n_cam = len(cls_per_cam)
        B = cls_per_cam[0].shape[0]
        proj_param = next(jepa.projector.parameters())
        per_cam_emb: list[Tensor] = []
        for cam_idx, cls in enumerate(cls_per_cam):
            packed = torch.zeros(B, n_cam * D_vit, device=cls.device, dtype=proj_param.dtype)
            packed[:, cam_idx * D_vit : (cam_idx + 1) * D_vit] = cls.to(proj_param.dtype)
            emb = jepa.projector(packed)  # (B, D)
            per_cam_emb.append(emb)
        # Stack cams along a "token" axis: (B, n_cam, D).
        return torch.stack(per_cam_emb, dim=1)
