import torch
import torch.nn.functional as F


def object_anchor_consistency(object_tokens: torch.Tensor, anchor_only: torch.Tensor) -> torch.Tensor:
    """Keep the learned object residual close to its externally supplied anchor semantics."""
    x = F.normalize(object_tokens.mean(dim=1), dim=-1)
    y = F.normalize(anchor_only.mean(dim=1).detach(), dim=-1)
    return (1.0 - (x * y).sum(dim=-1)).mean()


def sam2_object_alignment(
    object_features: torch.Tensor,
    object_labels: torch.Tensor,
    object_valid: torch.Tensor,
    class_text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Classify SAM2-region embeddings against structured B-CLIP text features."""
    feature_dim = class_text_features.shape[-1]
    features = object_features.reshape(-1, feature_dim)
    labels = object_labels.reshape(-1)
    valid = object_valid.reshape(-1) & (labels >= 0) & (labels < class_text_features.shape[0])
    if not torch.any(valid):
        return class_text_features.sum() * 0.0

    features = F.normalize(features[valid], dim=-1)
    text = F.normalize(class_text_features, dim=-1)
    logits = logit_scale * features @ text.t()
    return F.cross_entropy(logits, labels[valid])


def weak_anomaly_mil_loss(
    video_anomaly_logits: torch.Tensor,
    frame_anomaly_logits: torch.Tensor,
    video_anomaly_targets: torch.Tensor,
    sparsity_weight: float = 0.05,
) -> torch.Tensor:
    """Train frame anomaly evidence from video labels using top-k MIL pooling."""
    video_loss = F.binary_cross_entropy_with_logits(
        video_anomaly_logits,
        video_anomaly_targets.to(video_anomaly_logits.dtype),
    )
    positive = video_anomaly_targets > 0.5
    if positive.any() and sparsity_weight > 0:
        sparsity_loss = torch.sigmoid(frame_anomaly_logits[positive]).mean(dim=1).mean()
        video_loss = video_loss + sparsity_weight * sparsity_loss
    return video_loss


def hierarchical_anomaly_loss(
    video_anomaly_logits: torch.Tensor,
    frame_anomaly_logits: torch.Tensor,
    object_anomaly_scores: torch.Tensor,
    object_valid: torch.Tensor,
    frame_object_anomaly_scores: torch.Tensor,
    object_frame_gate: torch.Tensor,
    video_anomaly_targets: torch.Tensor,
    sparsity_weight: float = 0.05,
    normal_frame_weight: float = 0.1,
    normal_object_weight: float = 0.1,
    hierarchy_weight: float = 0.1,
) -> torch.Tensor:
    """Propagate video labels to frames and proposals without positive instance labels."""
    loss = weak_anomaly_mil_loss(
        video_anomaly_logits,
        frame_anomaly_logits,
        video_anomaly_targets,
        sparsity_weight=sparsity_weight,
    )
    normal_videos = video_anomaly_targets <= 0.5
    if normal_videos.any():
        frame_scores = torch.sigmoid(frame_anomaly_logits)
        loss = loss + normal_frame_weight * frame_scores[normal_videos].mean()

        valid = object_valid[normal_videos]
        scores = object_anomaly_scores[normal_videos] * valid.to(object_anomaly_scores.dtype)
        counts = valid.sum(dim=(1, 2)).clamp_min(1).to(scores.dtype)
        per_video_object_scores = scores.sum(dim=(1, 2)) / counts
        loss = loss + normal_object_weight * per_video_object_scores.mean()

    frames_with_objects = object_valid.any(dim=-1)
    if frames_with_objects.any():
        consistency = torch.abs(
            torch.sigmoid(frame_anomaly_logits) - frame_object_anomaly_scores
        )
        weighted_consistency = consistency * object_frame_gate * frames_with_objects.to(consistency.dtype)
        loss = loss + hierarchy_weight * (
            weighted_consistency.sum() / frames_with_objects.sum().clamp_min(1)
        )
    return loss
