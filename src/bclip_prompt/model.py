from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import object_anchor_consistency, sam2_object_alignment
from .prompt_learner import StructuredPromptLearner
from .video_branch import WeaklySupervisedVideoBranch


class StructuredBCLIP(nn.Module):
    def __init__(
        self,
        adapter,
        prompt_learner: StructuredPromptLearner,
        mode: str = "conditioned",
        freeze_bclip: bool = True,
        video_branch_weight: float = 1.0,
        anomaly_topk: int = 1,
        inference_frame_chunk_size: int = 32,
    ):
        super().__init__()
        if mode not in {"conditioned", "global"}:
            raise ValueError("mode must be 'conditioned' or 'global'")
        self.adapter = adapter
        self.prompt_learner = prompt_learner
        self.mode = mode
        self.video_branch_weight = float(video_branch_weight)
        self.inference_frame_chunk_size = max(1, int(inference_frame_chunk_size))
        self.video_branch = WeaklySupervisedVideoBranch(
            adapter.embed_dim,
            len(prompt_learner.classes),
            anomaly_topk=anomaly_topk,
        )

        if freeze_bclip:
            for p in self.adapter.model.parameters():
                p.requires_grad = False

    def encode_prompts(self):
        p = self.prompt_learner()
        text = self.adapter.encode_soft_text(p.embeddings, p.eot_positions)
        return p, F.normalize(text, dim=-1)

    def forward(
        self,
        frames: torch.Tensor,
        object_text_features: torch.Tensor | None = None,
        object_visual_features: torch.Tensor | None = None,
        segmentation_features: torch.Tensor | None = None,
        object_labels: torch.Tensor | None = None,
        object_valid: torch.Tensor | None = None,
    ):
        if frames.ndim == 5:
            batch_size, frame_count = frames.shape[:2]
            images = frames.flatten(0, 1)
        elif frames.ndim == 4:
            batch_size, frame_count = frames.shape[0], 1
            images = frames
        else:
            raise ValueError("frames must have shape [B,T,C,H,W] or [B,C,H,W]")

        prompt_batch, text_features = self.encode_prompts()  # [C,D]
        scale = self.adapter.logit_scale().clamp(max=100.0)
        flat_object_text = object_text_features.flatten(0, 1) if object_text_features is not None else None
        flat_object_visual = object_visual_features.flatten(0, 1) if object_visual_features is not None else None
        flat_object_valid = object_valid.flatten(0, 1) if object_valid is not None else None
        chunk_size = images.shape[0] if self.training else self.inference_frame_chunk_size
        frame_feature_chunks = []
        frame_logit_chunks = []

        for start in range(0, images.shape[0], chunk_size):
            end = min(start + chunk_size, images.shape[0])
            chunk_images = images[start:end].to(text_features.device, non_blocking=True)
            if self.mode == "global":
                image_features = F.normalize(self.adapter.encode_image_global(chunk_images), dim=-1)
                object_embeddings = [
                    features[start:end] for features in (flat_object_text, flat_object_visual)
                    if features is not None
                ]
                if object_embeddings and flat_object_valid is not None:
                    all_objects = torch.cat(object_embeddings, dim=1)
                    object_mask = flat_object_valid[start:end].repeat(1, len(object_embeddings))
                    mask = object_mask.to(all_objects.dtype).unsqueeze(-1)
                    pooled = (all_objects * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                    image_features = F.normalize(image_features + pooled, dim=-1)
                chunk_logits = scale * image_features @ text_features.t()
                chunk_frame_features = image_features
            else:
                patches = self.adapter.encode_image_patches(chunk_images)  # [frames,P,D]
                chunk_frame_features = F.normalize(patches.mean(dim=1), dim=-1)
                queries = text_features.unsqueeze(0).expand(chunk_images.shape[0], -1, -1)
                object_embeddings = [
                    features[start:end] for features in (flat_object_text, flat_object_visual)
                    if features is not None
                ]
                object_mask = None
                if object_embeddings:
                    object_queries = torch.cat(object_embeddings, dim=1)
                    if flat_object_valid is not None:
                        object_mask = flat_object_valid[start:end].repeat(1, len(object_embeddings))
                    queries = torch.cat((queries, object_queries), dim=1)
                conditioned = F.normalize(self.adapter.condition_patches(patches, queries), dim=-1)
                class_count = text_features.shape[0]
                class_conditioned = conditioned[:, :class_count]
                if object_mask is not None and conditioned.shape[1] > class_count:
                    mask = object_mask.to(conditioned.dtype).unsqueeze(-1)
                    object_context = (conditioned[:, class_count:] * mask).sum(dim=1)
                    object_context = object_context / mask.sum(dim=1).clamp_min(1.0)
                    has_objects = object_mask.any(dim=1, keepdim=True).unsqueeze(-1)
                    class_conditioned = F.normalize(
                        class_conditioned + object_context[:, None] * has_objects,
                        dim=-1,
                    )
                chunk_logits = scale * torch.einsum("bcd,cd->bc", class_conditioned, text_features)

            frame_feature_chunks.append(chunk_frame_features)
            frame_logit_chunks.append(chunk_logits)

        frame_features = torch.cat(frame_feature_chunks, dim=0)
        frame_logits = torch.cat(frame_logit_chunks, dim=0)

        bclip_logits = frame_logits.reshape(batch_size, frame_count, -1).mean(dim=1)

        if object_visual_features is None and object_text_features is None:
            object_visual_features = frame_features.new_zeros((batch_size, frame_count, 1, frame_features.shape[-1]))
            object_text_features = torch.zeros_like(object_visual_features)
            object_valid = torch.zeros(
                (batch_size, frame_count, 1), dtype=torch.bool, device=frame_features.device
            )
        elif object_visual_features is None:
            object_visual_features = torch.zeros_like(object_text_features)
        elif object_text_features is None:
            object_text_features = torch.zeros_like(object_visual_features)
        if segmentation_features is None:
            segmentation_features = frame_features.new_zeros((*object_visual_features.shape[:3], 5))
        if object_valid is None:
            object_valid = torch.ones(
                object_visual_features.shape[:3], dtype=torch.bool, device=frame_features.device
            )

        branch = self.video_branch(
            frame_features.reshape(batch_size, frame_count, -1),
            object_visual_features,
            object_text_features,
            segmentation_features,
            object_valid,
        )
        logits = (
            F.log_softmax(bclip_logits, dim=-1)
            + self.video_branch_weight * F.log_softmax(branch["logits"], dim=-1)
        )

        anchor_loss = object_anchor_consistency(
            prompt_batch.object_tokens, prompt_batch.object_anchor_only
        )
        if object_visual_features is not None and object_labels is not None and object_valid is not None:
            object_loss = sam2_object_alignment(
                object_visual_features,
                object_labels,
                object_valid,
                text_features,
                scale,
            )
        else:
            object_loss = text_features.sum() * 0.0
        return {
            "logits": logits,
            "bclip_logits": bclip_logits,
            "video_branch_logits": branch["logits"],
            "frame_attention": branch["frame_attention"],
            "object_attention": branch["object_attention"],
            "frame_anomaly_logits": branch["frame_anomaly_logits"],
            "frame_anomaly_scores": branch["frame_anomaly_scores"],
            "video_anomaly_logit": branch["video_anomaly_logit"],
            "video_anomaly_score": branch["video_anomaly_score"],
            "anchor_loss": anchor_loss,
            "object_loss": object_loss,
            "text_features": text_features,
        }
