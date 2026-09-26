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
from .dataset import VideoCSVClassificationDataset
from .losses import weak_anomaly_mil_loss
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
        video_branch_weight=cfg["model"].get("video_branch_weight", 1.0),
        anomaly_topk=cfg["model"].get("anomaly_topk", 1),
        inference_frame_chunk_size=cfg["data"].get("inference_frame_chunk_size", 32),
    ).to(device)
    return model


def make_optimizer(model: StructuredBCLIP, cfg: dict):
    tcfg = cfg["train"]
    groups = [{
        "params": [p for p in model.prompt_learner.parameters() if p.requires_grad],
        "lr": float(tcfg["lr_prompt"]),
    }]
    groups.append({
        "params": list(model.video_branch.parameters()),
        "lr": float(tcfg.get("lr_video_branch", tcfg["lr_prompt"])),
    })

    if tcfg.get("train_conditioner", False):
        block = getattr(model.adapter.model, "text_conditioned_patches_block", None)
        if block is None:
            raise AttributeError("train_conditioner=true but β-CLIP conditioner is unavailable")
        for p in block.parameters():
            p.requires_grad = True
        groups.append({"params": list(block.parameters()), "lr": float(tcfg["lr_conditioner"])})

    return torch.optim.AdamW(groups, weight_decay=float(tcfg.get("weight_decay", 1e-4)))


def build_video_dataset(
    cfg: dict,
    train: bool,
    all_frames: bool = False,
) -> VideoCSVClassificationDataset:
    dcfg = cfg["data"]
    scfg = cfg.get("sam2", {})
    csv_key = "train_csv" if train else "val_csv"
    return VideoCSVClassificationDataset(
        dcfg[csv_key],
        root=dcfg.get("root", "."),
        image_size=dcfg.get("image_size", 224),
        frames_per_video=None if all_frames else dcfg.get("frames_per_video", 8),
        cache_dir=scfg.get("cache_dir", "data/sam2_cache"),
        max_objects=scfg.get("max_objects", 8),
        object_dim=dcfg.get("object_dim", 512),
        require_cache=dcfg.get("require_sam2_cache", True),
        train=train,
    )


def make_anomaly_targets(labels: torch.Tensor, cfg: dict) -> torch.Tensor:
    anomaly_class_ids = cfg["model"].get("anomaly_class_ids", [1])
    if not anomaly_class_ids:
        raise ValueError("model.anomaly_class_ids must include at least one anomalous class")
    anomaly_ids = torch.as_tensor(anomaly_class_ids, device=labels.device)
    return torch.isin(labels, anomaly_ids).to(torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, cfg, return_predictions: bool = False):
    model.eval()
    n = correct = branch_correct = anomaly_correct = 0
    loss_sum = 0.0
    predictions = []
    branch_predictions = []
    video_anomaly_scores = []
    frame_anomaly_scores = []
    frame_indices_all = []
    anchor_w = float(cfg["train"].get("anchor_loss_weight", 0.0))
    object_w = float(cfg["train"].get("object_loss_weight", 0.1))
    anomaly_w = float(cfg["train"].get("anomaly_loss_weight", 1.0))
    sparsity_w = float(cfg["train"].get("anomaly_sparsity_weight", 0.05))
    for frames, labels, visual, text, segmentation, object_labels, valid, frame_indices in loader:
        labels = labels.to(device)
        visual = visual.to(device)
        text = text.to(device)
        segmentation = segmentation.to(device)
        object_labels = object_labels.to(device)
        valid = valid.to(device)
        out = model(
            frames,
            object_text_features=text,
            object_visual_features=visual,
            segmentation_features=segmentation,
            object_labels=object_labels,
            object_valid=valid,
        )
        loss = (
            F.cross_entropy(out["logits"], labels)
            + float(cfg["train"].get("video_branch_loss_weight", 1.0))
            * F.cross_entropy(out["video_branch_logits"], labels)
            + anomaly_w * weak_anomaly_mil_loss(
                out["video_anomaly_logit"],
                out["frame_anomaly_logits"],
                make_anomaly_targets(labels, cfg),
                sparsity_weight=sparsity_w,
            )
            + anchor_w * out["anchor_loss"]
            + object_w * out["object_loss"]
        )
        loss_sum += loss.item() * labels.numel()
        correct += (out["logits"].argmax(dim=1) == labels).sum().item()
        branch_correct += (out["video_branch_logits"].argmax(dim=1) == labels).sum().item()
        anomaly_targets = make_anomaly_targets(labels, cfg).bool()
        anomaly_correct += ((out["video_anomaly_score"] >= 0.5) == anomaly_targets).sum().item()
        if return_predictions:
            predictions.extend(out["logits"].argmax(dim=1).cpu().tolist())
            branch_predictions.extend(out["video_branch_logits"].argmax(dim=1).cpu().tolist())
            video_anomaly_scores.extend(out["video_anomaly_score"].cpu().tolist())
            frame_anomaly_scores.extend(out["frame_anomaly_scores"].cpu().tolist())
            frame_indices_all.extend(frame_indices.tolist())
        n += labels.numel()
    metrics = {
        "loss": loss_sum / max(n, 1),
        "acc": correct / max(n, 1),
        "branch_acc": branch_correct / max(n, 1),
        "anomaly_acc": anomaly_correct / max(n, 1),
    }
    if return_predictions:
        metrics["predictions"] = predictions
        metrics["branch_predictions"] = branch_predictions
        metrics["video_anomaly_scores"] = video_anomaly_scores
        metrics["frame_anomaly_scores"] = frame_anomaly_scores
        metrics["frame_indices"] = frame_indices_all
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "cuda"))
    print(f"device={device}")

    dcfg = cfg["data"]
    train_ds = build_video_dataset(cfg, train=True)
    val_ds = build_video_dataset(cfg, train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=dcfg.get("num_workers", 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=dcfg.get("num_workers", 4), pin_memory=True)

    model = build_model(cfg, device)
    optimizer = make_optimizer(model, cfg)
    amp_enabled = bool(cfg["train"].get("amp", True) and device.type == "cuda")
    scaler = GradScaler(enabled=amp_enabled)
    anchor_w = float(cfg["train"].get("anchor_loss_weight", 0.0))
    object_w = float(cfg["train"].get("object_loss_weight", 0.1))
    out_dir = Path(cfg["train"].get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        running = 0.0
        seen = 0
        bar = tqdm(train_loader, desc=f"epoch {epoch+1}")
        for frames, labels, visual, text, segmentation, object_labels, valid, _frame_indices in bar:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            visual = visual.to(device, non_blocking=True)
            text = text.to(device, non_blocking=True)
            segmentation = segmentation.to(device, non_blocking=True)
            object_labels = object_labels.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                out = model(
                    frames,
                    object_text_features=text,
                    object_visual_features=visual,
                    segmentation_features=segmentation,
                    object_labels=object_labels,
                    object_valid=valid,
                )
                cls_loss = F.cross_entropy(out["logits"], labels)
                branch_loss = F.cross_entropy(out["video_branch_logits"], labels)
                anomaly_loss = weak_anomaly_mil_loss(
                    out["video_anomaly_logit"],
                    out["frame_anomaly_logits"],
                    make_anomaly_targets(labels, cfg),
                    sparsity_weight=float(cfg["train"].get("anomaly_sparsity_weight", 0.05)),
                )
                loss = (
                    cls_loss
                    + float(cfg["train"].get("video_branch_loss_weight", 1.0)) * branch_loss
                    + float(cfg["train"].get("anomaly_loss_weight", 1.0)) * anomaly_loss
                    + anchor_w * out["anchor_loss"]
                    + object_w * out["object_loss"]
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * labels.numel()
            seen += labels.numel()
            bar.set_postfix(
                loss=f"{running/max(seen,1):.4f}",
                obj=f"{out['object_loss'].item():.4f}",
            )

        metrics = evaluate(model, val_loader, device, cfg)
        print(
            f"val loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} "
            f"branch_acc={metrics['branch_acc']:.4f} anomaly_acc={metrics['anomaly_acc']:.4f}"
        )
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, best_acc, cfg)
        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, best_acc, cfg)
            print(f"new best acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
