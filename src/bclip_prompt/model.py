from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import object_anchor_consistency
from .prompt_learner import StructuredPromptLearner


class StructuredBCLIP(nn.Module):
    def __init__(self, adapter, prompt_learner: StructuredPromptLearner, mode: str = "conditioned", freeze_bclip: bool = True):
        super().__init__()
        if mode not in {"conditioned", "global"}:
            raise ValueError("mode must be 'conditioned' or 'global'")
        self.adapter = adapter
        self.prompt_learner = prompt_learner
        self.mode = mode

        if freeze_bclip:
            for p in self.adapter.model.parameters():
                p.requires_grad = False

    def encode_prompts(self):
        p = self.prompt_learner()
        text = self.adapter.encode_soft_text(p.embeddings, p.eot_positions)
        return p, F.normalize(text, dim=-1)

    def forward(self, images: torch.Tensor):
        prompt_batch, text_features = self.encode_prompts()  # [C,D]
        scale = self.adapter.logit_scale().clamp(max=100.0)

        if self.mode == "global":
            image_features = F.normalize(self.adapter.encode_image_global(images), dim=-1)
            logits = scale * image_features @ text_features.t()
        else:
            patches = self.adapter.encode_image_patches(images)  # [B,P,D]
            queries = text_features.unsqueeze(0).expand(images.shape[0], -1, -1)  # [B,C,D]
            conditioned = F.normalize(self.adapter.condition_patches(patches, queries), dim=-1)
            logits = scale * torch.einsum("bcd,cd->bc", conditioned, text_features)

        anchor_loss = object_anchor_consistency(
            prompt_batch.object_tokens, prompt_batch.object_anchor_only
        )
        return {"logits": logits, "anchor_loss": anchor_loss, "text_features": text_features}
