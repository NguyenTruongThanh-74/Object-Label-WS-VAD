from __future__ import annotations

import torch
import torch.nn as nn


class WeaklySupervisedVideoBranch(nn.Module):
    """Aggregate frame and segmented-object features into video-level logits."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        segmentation_dim: int = 5,
        dropout: float = 0.1,
        anomaly_topk: int = 1,
    ):
        super().__init__()
        self.object_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3 + segmentation_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.object_attention = nn.Linear(embed_dim, 1, bias=False)
        self.frame_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.frame_attention = nn.Linear(embed_dim, 1, bias=False)
        self.anomaly_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        self.anomaly_topk = max(1, int(anomaly_topk))

    def forward(
        self,
        frame_features: torch.Tensor,
        object_visual_features: torch.Tensor,
        object_text_features: torch.Tensor,
        segmentation_features: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, frame_count, object_count, _ = object_visual_features.shape
        frame_context = frame_features.unsqueeze(2).expand(-1, -1, object_count, -1)
        object_tokens = self.object_fusion(torch.cat(
            (
                frame_context,
                object_visual_features,
                object_text_features,
                segmentation_features,
            ),
            dim=-1,
        ))

        object_scores = self.object_attention(object_tokens).squeeze(-1)
        object_scores = object_scores.masked_fill(~object_valid, -1e4)
        object_weights = torch.softmax(object_scores, dim=-1) * object_valid.to(object_scores.dtype)
        object_weights = object_weights / object_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        object_context = torch.sum(object_tokens * object_weights.unsqueeze(-1), dim=2)

        frame_tokens = self.frame_fusion(torch.cat((frame_features, object_context), dim=-1))
        frame_weights = torch.softmax(self.frame_attention(frame_tokens).squeeze(-1), dim=1)
        video_features = torch.sum(frame_tokens * frame_weights.unsqueeze(-1), dim=1)
        temporal_context = frame_tokens.mean(dim=1, keepdim=True)
        frame_deviation = torch.abs(frame_tokens - temporal_context)
        frame_anomaly_logits = self.anomaly_head(
            torch.cat((frame_tokens, frame_deviation), dim=-1)
        ).squeeze(-1)
        topk = min(self.anomaly_topk, frame_count)
        video_anomaly_logit = frame_anomaly_logits.topk(topk, dim=1).values.mean(dim=1)
        return {
            "logits": self.classifier(video_features),
            "frame_attention": frame_weights,
            "object_attention": object_weights,
            "frame_anomaly_logits": frame_anomaly_logits,
            "frame_anomaly_scores": torch.sigmoid(frame_anomaly_logits),
            "video_anomaly_logit": video_anomaly_logit,
            "video_anomaly_score": torch.sigmoid(video_anomaly_logit),
        }