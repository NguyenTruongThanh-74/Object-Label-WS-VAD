from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .bclip_adapter import BClipAdapter
from .dataset import ImageCSVClassificationDataset
from .model import StructuredBCLIP
from .prompt_learner import StructuredPromptLearner
from .utils import load_json, load_yaml, resolve_device, save_checkpoint, seed_everything


def build_model(cfg: dict, device: torch.device) -> StructuredBCLIP:
    classes = load_json(cfg["data"]["classes_json"])
    anchors = torch.from_numpy(np.load(cfg["data"]["anchors_npy"]))
    if len(classes) != anchors.shape[0]:
        raise ValueError(f"{len(classes)} classes but anchors has {anchors.shape[0]} rows")

    adapter = BClipAdapter(cfg["bclip"], device)
    prompt = StructuredPromptLearner(adapter, classes, anchors, cfg["prompt"])
    model = StructuredBCLIP(
        adapter,
        prompt,
        mode=cfg["model"].get("mode", "conditioned"),
        freeze_bclip=cfg["model"].get("freeze_bclip", True),
    ).to(device)
    return model


def make_optimizer(model: StructuredBCLIP, cfg: dict):
    tcfg = cfg["train"]
    groups = [{
        "params": [p for p in model.prompt_learner.parameters() if p.requires_grad],
        "lr": float(tcfg["lr_prompt"]),
    }]

    if tcfg.get("train_conditioner", False):
        block = getattr(model.adapter.model, "text_conditioned_patches_block", None)
        if block is None:
            raise AttributeError("train_conditioner=true but β-CLIP conditioner is unavailable")
        for p in block.parameters():
            p.requires_grad = True
        groups.append({"params": list(block.parameters()), "lr": float(tcfg["lr_conditioner"])})

    return torch.optim.AdamW(groups, weight_decay=float(tcfg.get("weight_decay", 1e-4)))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = correct = 0
    loss_sum = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        loss = F.cross_entropy(out["logits"], labels)
        loss_sum += loss.item() * labels.numel()
        correct += (out["logits"].argmax(dim=1) == labels).sum().item()
        n += labels.numel()
    return {"loss": loss_sum / max(n, 1), "acc": correct / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "cuda"))
    print(f"device={device}")

    dcfg = cfg["data"]
    train_ds = ImageCSVClassificationDataset(dcfg["train_csv"], dcfg.get("root", "."), dcfg.get("image_size", 224), train=True)
    val_ds = ImageCSVClassificationDataset(dcfg["val_csv"], dcfg.get("root", "."), dcfg.get("image_size", 224), train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=dcfg.get("num_workers", 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=dcfg.get("num_workers", 4), pin_memory=True)

    model = build_model(cfg, device)
    optimizer = make_optimizer(model, cfg)
    amp_enabled = bool(cfg["train"].get("amp", True) and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)
    anchor_w = float(cfg["train"].get("anchor_loss_weight", 0.0))
    out_dir = Path(cfg["train"].get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        running = 0.0
        seen = 0
        bar = tqdm(train_loader, desc=f"epoch {epoch+1}")
        for images, labels in bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                out = model(images)
                cls_loss = F.cross_entropy(out["logits"], labels)
                loss = cls_loss + anchor_w * out["anchor_loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * labels.numel()
            seen += labels.numel()
            bar.set_postfix(loss=f"{running/max(seen,1):.4f}", anchor=f"{out['anchor_loss'].item():.4f}")

        metrics = evaluate(model, val_loader, device)
        print(f"val loss={metrics['loss']:.4f} acc={metrics['acc']:.4f}")
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, best_acc, cfg)
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, best_acc, cfg)
            print(f"new best acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
