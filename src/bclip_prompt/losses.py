import torch
import torch.nn.functional as F


def object_anchor_consistency(object_tokens: torch.Tensor, anchor_only: torch.Tensor) -> torch.Tensor:
    """Keep the learned object residual close to its externally supplied anchor semantics."""
    x = F.normalize(object_tokens.mean(dim=1), dim=-1)
    y = F.normalize(anchor_only.mean(dim=1).detach(), dim=-1)
    return (1.0 - (x * y).sum(dim=-1)).mean()
