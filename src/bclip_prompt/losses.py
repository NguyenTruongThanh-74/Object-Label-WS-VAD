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
