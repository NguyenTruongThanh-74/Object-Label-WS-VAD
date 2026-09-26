from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import IMAGE_SUFFIXES
from .train import build_model, build_video_dataset, evaluate
from .utils import load_json, load_yaml, resolve_device, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--predictions", default="outputs/video_predictions.csv")
    ap.add_argument("--frame-scores", default="outputs/frame_anomaly_scores.csv")
    ap.add_argument("--anomaly-threshold", type=float, default=0.5)
    ap.add_argument("--all-frames", action="store_true", help="Score every frame instead of sampled frames")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "cuda"))
    model = build_model(cfg, device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    print(f"loaded research checkpoint; missing={len(missing)} unexpected={len(unexpected)}")

    ds = build_video_dataset(cfg, train=False, all_frames=args.all_frames)
    batch_size = 1 if args.all_frames else cfg["train"]["batch_size"]
    num_workers = 0 if args.all_frames else cfg["data"].get("num_workers", 4)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    metrics = evaluate(model, loader, device, cfg, return_predictions=True)
    predictions_path = Path(args.predictions)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    classes = {int(item["id"]): item["name"] for item in load_json(cfg["data"]["classes_json"])}
    anomaly_class_ids = set(cfg["model"].get("anomaly_class_ids", [1]))
    with predictions_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "video", "true_label", "predicted_label", "predicted_class",
            "attention_branch_label", "attention_branch_class", "video_anomaly_score",
            "true_is_anomaly", "predicted_is_anomaly",
        ])
        writer.writeheader()
        for (video, true_label), predicted, branch_predicted, anomaly_score in zip(
            ds.rows, metrics["predictions"], metrics["branch_predictions"],
            metrics["video_anomaly_scores"],
        ):
            writer.writerow({
                "video": video,
                "true_label": true_label,
                "predicted_label": predicted,
                "predicted_class": classes[predicted],
                "attention_branch_label": branch_predicted,
                "attention_branch_class": classes[branch_predicted],
                "video_anomaly_score": anomaly_score,
                "true_is_anomaly": int(true_label in anomaly_class_ids),
                "predicted_is_anomaly": int(anomaly_score >= args.anomaly_threshold),
            })
    frame_scores_path = Path(args.frame_scores)
    frame_scores_path.parent.mkdir(parents=True, exist_ok=True)
    with frame_scores_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "video", "true_video_label", "frame_index", "frame_file", "video_anomaly_score",
            "frame_anomaly_score", "predicted_is_anomalous_frame",
        ])
        writer.writeheader()
        root = Path(cfg["data"].get("root", "."))
        for video_index, (video, true_label) in enumerate(ds.rows):
            video_path = Path(video)
            if not video_path.is_absolute():
                video_path = root / video_path
            frame_paths = sorted(
                path for path in video_path.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            for frame_index, score in zip(
                metrics["frame_indices"][video_index],
                metrics["frame_anomaly_scores"][video_index],
            ):
                writer.writerow({
                    "video": video,
                    "true_video_label": true_label,
                    "frame_index": frame_index,
                    "frame_file": frame_paths[frame_index].name,
                    "video_anomaly_score": metrics["video_anomaly_scores"][video_index],
                    "frame_anomaly_score": score,
                    "predicted_is_anomalous_frame": int(score >= args.anomaly_threshold),
                })
    print(
        f"accuracy={metrics['acc']:.6f} branch_accuracy={metrics['branch_acc']:.6f} "
        f"anomaly_accuracy={metrics['anomaly_acc']:.6f} loss={metrics['loss']:.6f} "
        f"predictions={predictions_path} frame_scores={frame_scores_path}"
    )


if __name__ == "__main__":
    main()
