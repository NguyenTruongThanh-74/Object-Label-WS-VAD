from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PromptBatch:
    embeddings: torch.Tensor      # [C, L, text_width]
    eot_positions: torch.Tensor   # [C]
    object_tokens: torch.Tensor   # [C, n_obj, text_width]
    object_anchor_only: torch.Tensor  # [C, n_obj, text_width]


class StructuredPromptLearner(nn.Module):
    """Build continuous prompts in the exact order:

    objects + actions + place + and where

    Object learnable tokens are anchored by one external vector per class.
    """

    def __init__(self, adapter, classes: List[dict], anchors: torch.Tensor, cfg: dict):
        super().__init__()
        self.adapter = adapter
        self.classes = sorted(classes, key=lambda x: int(x["id"]))
        self.register_buffer("anchors", anchors.float(), persistent=True)

        width = adapter.text_width
        anchor_dim = int(cfg.get("anchor_dim", anchors.shape[-1]))
        if anchors.shape[-1] != anchor_dim:
            raise ValueError(f"anchors dim={anchors.shape[-1]} but prompt.anchor_dim={anchor_dim}")

        self.n_obj = int(cfg.get("object_tokens", 1))
        self.n_action = int(cfg.get("action_tokens", 2))
        self.n_place = int(cfg.get("place_tokens", 2))
        self.n_where = int(cfg.get("where_tokens", 2))
        init_std = float(cfg.get("init_std", 0.02))

        self.anchor_proj = nn.Linear(anchor_dim, width, bias=True)
        self.object_residual = nn.Parameter(torch.empty(self.n_obj, width))
        self.action_ctx = nn.Parameter(torch.empty(self.n_action, width))
        self.place_ctx = nn.Parameter(torch.empty(self.n_place, width))
        self.where_ctx = nn.Parameter(torch.empty(self.n_where, width))
        self.object_ln = nn.LayerNorm(width)
        self.use_object_residual = bool(cfg.get("use_object_residual", True))
        self.normalize_anchor = bool(cfg.get("normalize_anchor_before_projection", True))

        init_scale = float(cfg.get("anchor_scale_init", 1.0))
        if cfg.get("learnable_anchor_scale", True):
            self.anchor_scale = nn.Parameter(torch.tensor(init_scale))
        else:
            self.register_buffer("anchor_scale", torch.tensor(init_scale), persistent=True)

        nn.init.normal_(self.object_residual, std=init_std)
        nn.init.normal_(self.action_ctx, std=init_std)
        nn.init.normal_(self.place_ctx, std=init_std)
        nn.init.normal_(self.where_ctx, std=init_std)
        nn.init.xavier_uniform_(self.anchor_proj.weight)
        nn.init.zeros_(self.anchor_proj.bias)

        self._fixed_tokens = self._prepare_fixed_tokens()

    def _ids(self, s: str) -> torch.Tensor:
        ids = self.adapter.token_ids(s)
        return torch.tensor(ids, dtype=torch.long)

    def _prepare_fixed_tokens(self) -> Dict[str, torch.Tensor]:
        fixed = {
            "objects": self._ids("objects:"),
            "actions": self._ids("actions:"),
            "place": self._ids("place:"),
            "where": self._ids("and where:"),
        }
        for c in self.classes:
            cid = int(c["id"])
            for key in ("object", "action", "place", "where"):
                fixed[f"c{cid}_{key}"] = self._ids(str(c[key]))
        return fixed

    def _embed_ids(self, ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        if ids.numel() == 0:
            return torch.empty(0, self.adapter.text_width, device=device)
        ids = ids.to(device)
        # Descriptor words and structural labels stay frozen even if B-CLIP is trainable.
        with torch.no_grad():
            return self.adapter.embed_token_ids(ids).detach()

    def _object_tokens(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.anchors
        if self.normalize_anchor:
            a = F.normalize(a, dim=-1)
        base = self.anchor_proj(a) * self.anchor_scale
        base = base[:, None, :].expand(-1, self.n_obj, -1)
        anchor_only = self.object_ln(base)
        if self.use_object_residual:
            obj = self.object_ln(base + self.object_residual[None])
        else:
            obj = anchor_only
        return obj, anchor_only

    def forward(self) -> PromptBatch:
        device = self.object_residual.device
        obj_all, anchor_only_all = self._object_tokens()
        sequences = []
        eot_positions = []

        sot = self._embed_ids(torch.tensor([self.adapter.sot_id]), device)
        eot = self._embed_ids(torch.tensor([self.adapter.eot_id]), device)

        for c in self.classes:
            cid = int(c["id"])
            chunks = [
                sot,
                self._embed_ids(self._fixed_tokens["objects"], device),
                obj_all[cid],
                self._embed_ids(self._fixed_tokens[f"c{cid}_object"], device),
                self._embed_ids(self._fixed_tokens["actions"], device),
                self.action_ctx,
                self._embed_ids(self._fixed_tokens[f"c{cid}_action"], device),
                self._embed_ids(self._fixed_tokens["place"], device),
                self.place_ctx,
                self._embed_ids(self._fixed_tokens[f"c{cid}_place"], device),
                self._embed_ids(self._fixed_tokens["where"], device),
                self.where_ctx,
                self._embed_ids(self._fixed_tokens[f"c{cid}_where"], device),
                eot,
            ]
            seq = torch.cat(chunks, dim=0)
            if seq.shape[0] > self.adapter.context_length:
                raise ValueError(
                    f"Prompt for class {cid} has {seq.shape[0]} tokens, exceeding "
                    f"β-CLIP context length {self.adapter.context_length}. Shorten descriptors "
                    "or reduce learnable prompt tokens."
                )
            eot_positions.append(seq.shape[0] - 1)
            pad_len = self.adapter.context_length - seq.shape[0]
            if pad_len:
                seq = torch.cat([seq, torch.zeros(pad_len, seq.shape[-1], device=device, dtype=seq.dtype)], dim=0)
            sequences.append(seq)

        return PromptBatch(
            embeddings=torch.stack(sequences, dim=0),
            eot_positions=torch.tensor(eot_positions, device=device, dtype=torch.long),
            object_tokens=obj_all,
            object_anchor_only=anchor_only_all,
        )
