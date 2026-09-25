from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


class BClipAdapter(torch.nn.Module):
    """Thin adapter around the official β-CLIP repository.

    It intentionally depends on the upstream repository rather than copying it.
    """

    def __init__(self, cfg: dict, device: torch.device):
        super().__init__()
        repo = Path(cfg["repo"]).expanduser().resolve()
        if not repo.exists():
            raise FileNotFoundError(
                f"β-CLIP repo not found: {repo}. Run `bash scripts/setup_bclip.sh` first."
            )
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        models = importlib.import_module("models_tome")
        tokenizer_mod = importlib.import_module("tokenizer")
        self.tokenizer = tokenizer_mod.SimpleTokenizer()

        model_name = cfg.get("model_name", "CLIP_VITB16_OPENAI")
        factory = getattr(models, model_name)
        self.model = factory(
            attn_fn=cfg.get("attn_fn", "softmax"),
            attn_fn_alpha=float(cfg.get("attn_fn_alpha", 1.0)),
            global_pool=cfg.get("global_pool", ""),
            text_conditioning_mode=cfg.get("text_conditioning_mode", "attn_pooling_mlp"),
            use_text_conditioned_patches=True,
            use_text_eos=True,
            use_text_concepts=False,
            use_text_tokens=False,
            use_text_conditioned_cls=cfg.get("use_text_conditioned_cls", False),
            context_length=cfg.get("context_length", 248),
        )

        ckpt = cfg.get("checkpoint", "")
        if ckpt:
            self._load_checkpoint(ckpt)
        self.model.to(device)

    def _load_checkpoint(self, path: str):
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        raw = torch.load(p, map_location="cpu", weights_only=False)

        if p.suffix == ".pt" and hasattr(raw, "state_dict"):
            state = raw.state_dict()
            if hasattr(self.model, "convert_state_dict_from_openai"):
                state = self.model.convert_state_dict_from_openai(state)
            if hasattr(self.model, "convert_state_dict"):
                state = self.model.convert_state_dict(state)
        else:
            if isinstance(raw, dict) and "state_dict" in raw:
                state = raw["state_dict"]
            elif isinstance(raw, dict) and "model" in raw:
                state = raw["model"]
            else:
                state = raw
            clean = {}
            for k, v in state.items():
                clean[k.removeprefix("module.")] = v
            state = clean

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(f"Loaded β-CLIP checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

    @property
    def text_width(self) -> int:
        return int(self.model.token_embedding.weight.shape[1])

    @property
    def embed_dim(self) -> int:
        return int(self.model.text_projection.shape[1])

    @property
    def context_length(self) -> int:
        return int(self.model.context_length)

    def token_ids(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    @property
    def sot_id(self) -> int:
        return int(self.tokenizer.encoder["<|startoftext|>"])

    @property
    def eot_id(self) -> int:
        return int(self.tokenizer.encoder["<|endoftext|>"])

    def embed_token_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.token_embedding(ids)

    def encode_soft_text(self, embeddings: torch.Tensor, eot_positions: torch.Tensor) -> torch.Tensor:
        """Run β-CLIP text transformer from continuous token embeddings."""
        max_len = embeddings.shape[1]
        x = embeddings
        if self.model.context_length == 248 and hasattr(self.model, "positional_embedding_res"):
            p1 = self.model.positional_embedding[:max_len].to(x.device)
            p2 = self.model.positional_embedding_res[:max_len].to(x.device)
            m1 = self.model.mask1[:max_len].to(x.device)
            m2 = self.model.mask2[:max_len].to(x.device)
            x = x + (p1 * m1).to(x.dtype) + (p2 * m2).to(x.dtype)
        else:
            x = x + self.model.positional_embedding[:max_len].to(x.device, dtype=x.dtype)
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x)
        row = torch.arange(x.shape[0], device=x.device)
        x = x[row, eot_positions]
        return x @ self.model.text_projection

    def encode_image_global(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.model.encode_image(images)
        if feat.dim() == 3:
            feat = feat[:, 0]
        return feat

    def encode_image_patches(self, images: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.model, "encode_image_by_block"):
            raise AttributeError("This β-CLIP model does not expose encode_image_by_block")
        feat = self.model.encode_image_by_block(images)
        if not getattr(self.model, "use_text_conditioned_cls", False):
            feat = feat[:, 1:]
        return F.normalize(feat, dim=-1)

    def condition_patches(self, patch_features: torch.Tensor, text_queries: torch.Tensor) -> torch.Tensor:
        """patch_features: [B,P,D], text_queries: [B,K,D] -> [B,K,D]."""
        block = getattr(self.model, "text_conditioned_patches_block", None)
        if block is None:
            raise AttributeError("β-CLIP text_conditioned_patches_block is unavailable")
        out, _ = block(patch_features, F.normalize(text_queries, dim=-1), need_weights=False)
        return out

    def logit_scale(self) -> torch.Tensor:
        return self.model.logit_scale.exp()
