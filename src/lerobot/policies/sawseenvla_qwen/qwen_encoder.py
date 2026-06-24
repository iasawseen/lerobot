"""QwenEncoder: load Qwen3.5-0.8B as a black-box VL encoder.

Public surface:
  - `encode(images, task_strings)` returns per-layer hidden states selected by config
    plus an attention mask over the prefix sequence.
  - `vlm_hidden_size` exposes Qwen's text hidden size for projection wiring.

We deliberately do NOT touch Qwen's internals — no manual layer walks, no per-layer
K/V extraction. Hidden states are read via `output_hidden_states=True` and re-projected
by the expert.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class QwenEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        image_size: int = 256,
        freeze: bool = True,
        load_weights: bool = True,
        tokenizer_max_length: int = 48,
        pad_language_to: str = "max_length",
    ):
        super().__init__()
        from transformers import AutoConfig, AutoProcessor

        self.model_name = model_name
        self.image_size = image_size
        self.tokenizer_max_length = tokenizer_max_length
        self.pad_language_to = pad_language_to

        cfg = AutoConfig.from_pretrained(model_name)
        # Qwen3_5Config has text_config + vision_config sub-configs.
        text_cfg = getattr(cfg, "text_config", cfg)
        self.vlm_hidden_size = text_cfg.hidden_size
        self.num_hidden_layers = text_cfg.num_hidden_layers

        self.processor = AutoProcessor.from_pretrained(model_name)

        # Load Qwen3.5-0.8B as-published. Use ImageTextToText auto-class so the
        # vision tower is wired up.
        from transformers import AutoModelForImageTextToText
        if load_weights:
            self.qwen = AutoModelForImageTextToText.from_pretrained(
                model_name, dtype=torch.bfloat16
            )
        else:
            from transformers import AutoModel
            self.qwen = AutoModel.from_config(cfg)

        if freeze:
            for p in self.qwen.parameters():
                p.requires_grad_(False)
            self.qwen.eval()
        self.freeze = freeze

    def _to_pil_compatible(self, image_tensor: Tensor) -> Tensor:
        """Convert float [0, 1] (B, C, H, W) into uint8 (B, H, W, C) on CPU for the processor."""
        if image_tensor.ndim == 4 and image_tensor.shape[1] == 3:
            t = image_tensor
        else:
            raise ValueError(f"Expected (B, 3, H, W), got {image_tensor.shape}")
        if image_tensor.dtype.is_floating_point:
            # SawSeenVLA pre-normalizes to [-1, 1]; revert here to [0, 1] then 0-255.
            # We accept either [0, 1] or [-1, 1] inputs and detect by range.
            tmin = t.min().item()
            if tmin < -0.01:
                t = (t + 1.0) * 0.5
            t = t.clamp(0.0, 1.0)
            t = (t * 255.0).round().to(torch.uint8)
        # Resize to image_size×image_size via bilinear (Qwen processor's smart_resize
        # would do this on PIL — we do it pre-conversion for speed)
        if t.shape[-1] != self.image_size or t.shape[-2] != self.image_size:
            t = F.interpolate(t.float(), size=(self.image_size, self.image_size), mode="bilinear",
                              align_corners=False).round().to(torch.uint8)
        return t.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C) uint8 numpy

    def _build_messages(self, task: str, n_images: int) -> list[dict]:
        """Qwen3-VL chat-template message with N image placeholders + user task text."""
        content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": task}]
        return [{"role": "user", "content": content}]

    def _processor_inputs(self, images_per_cam: list[Tensor], tasks: list[str]) -> dict:
        """Call the Qwen processor on the (images, text) batch.

        images_per_cam: list[Tensor (B, 3, H, W)] — one per camera.
        tasks: list[str] of length B.
        """
        B = images_per_cam[0].shape[0]
        n_cams = len(images_per_cam)

        # Stack per-sample image lists: outer batch, inner cams.
        # processor expects list[list[PIL.Image]] or numpy-array equivalents.
        per_sample_imgs = []
        cam_uint8 = [self._to_pil_compatible(im) for im in images_per_cam]  # list[(B,H,W,C)]
        for b in range(B):
            per_sample_imgs.append([cam_uint8[c][b] for c in range(n_cams)])

        # Build chat-templated text per sample
        texts = []
        for task in tasks:
            msgs = self._build_messages(task, n_cams)
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)

        # Image placeholder tokens (<|image_pad|>) are inserted into input_ids by the
        # processor — at 256×256 / patch_size=16 / spatial_merge_size=2 → 64 tokens/cam
        # for Qwen3-VL. With 2 cameras + task text + special tokens, the prefix is
        # ~130-300 tokens; fixed-length truncation would chop image placeholders. We
        # always pad to the longest sequence in the batch.
        kwargs = dict(text=texts, images=per_sample_imgs, return_tensors="pt", padding=True)
        return self.processor(**kwargs)

    def forward(
        self,
        images_per_cam: list[Tensor],
        tasks: list[str],
        anchor_layer_indices: tuple[int, ...],
    ) -> tuple[list[Tensor], Tensor]:
        """Run Qwen forward and return selected hidden states + attention mask.

        Returns:
          anchors: list of (B, L_prefix, vlm_hidden) tensors, one per anchor index.
          attention_mask: (B, L_prefix) bool/long tensor (True = valid token).
        """
        device = images_per_cam[0].device
        inputs = self._processor_inputs(images_per_cam, tasks)
        # Move to device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Optional: under freeze=True, Qwen forward runs under inference_mode to skip
        # autograd bookkeeping. Hidden states still flow gradients for the expert via
        # the re-projection inside the expert's cross-attention layers (the tensor is
        # part of the graph for the *expert's* parameters even though it's not
        # differentiable w.r.t. Qwen's parameters).
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            out = self.qwen(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        # out.hidden_states is a tuple (len = num_hidden_layers + 1).
        # Index 0 = input embeddings; indices 1..N = layer 0..N-1 outputs.
        hs = out.hidden_states
        max_idx = len(hs) - 1
        for idx in anchor_layer_indices:
            if idx < 0 or idx > max_idx:
                raise ValueError(
                    f"Anchor index {idx} out of range [0, {max_idx}] for Qwen with "
                    f"{self.num_hidden_layers} hidden layers."
                )
        anchors = [hs[i].float() for i in anchor_layer_indices]

        attn_mask = inputs["attention_mask"]
        return anchors, attn_mask
