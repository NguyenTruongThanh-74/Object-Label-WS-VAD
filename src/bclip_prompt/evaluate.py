from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .dataset import ImageCSVClassificationDataset
from .train import build_model, evaluate
from .utils import load_yaml, resolve_device, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "cuda"))
    model = build_model(cfg, device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    print(f"loaded research checkpoint; missing={len(missing)} unexpected={len(unexpected)}")

    dcfg = cfg["data"]
    ds = ImageCSVClassificationDataset(dcfg["val_csv"], dcfg.get("root", "."), dcfg.get("image_size", 224), train=False)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=dcfg.get("num_workers", 4))
    metrics = evaluate(model, loader, device)
    print(f"accuracy={metrics['acc']:.6f} loss={metrics['loss']:.6f}")


if __name__ == "__main__":
    main()
