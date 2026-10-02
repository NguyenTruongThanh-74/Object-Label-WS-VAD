from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .dataset import IMAGE_SUFFIXES, video_cache_key
from .train import build_model, build_video_dataset, evaluate
from .utils import load_json, load_yaml, resolve_device, seed_everything


def compose_spatial_anomaly_map(
    masks: list[np.ndarray],
    scores: list[float],
    output_size: tuple[int, int],
) -> np.ndarray:
    """Combine scored object masks into a pixelwise noisy-OR anomaly map."""
    width, height = output_size
    survival = np.ones((height, width), dtype=np.float32)
    for mask, score in zip(masks, scores):
        binary_mask = np.asarray(mask) > 0
        if binary_mask.shape != (height, width):
            resized = Image.fromarray(binary_mask.astype(np.uint8) * 255).resize(
                output_size, resample=Image.Resampling.NEAREST
            )
            binary_mask = np.asarray(resized) > 0
        survival *= 1.0 - np.clip(float(score), 0.0, 1.0) * binary_mask
    return 1.0 - survival


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--predictions", default="outputs/video_predictions.csv")
    ap.add_argument("--frame-scores", default="outputs/frame_anomaly_scores.csv")
    ap.add_argument("--object-scores", default="outputs/object_anomaly_scores.csv")
    ap.add_argument("--spatial-maps", default="outputs/spatial_anomaly_maps")
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
    object_scores_path = Path(args.object_scores)
    object_scores_path.parent.mkdir(parents=True, exist_ok=True)
    spatial_maps_root = Path(args.spatial_maps)
    spatial_maps_root.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["data"].get("root", "."))
    segmentation_root = Path(cfg.get("sam2", {}).get("segmentation_dir", "data/sam2_masks"))
    with frame_scores_path.open("w", encoding="utf-8", newline="") as frame_file, \
            object_scores_path.open("w", encoding="utf-8", newline="") as object_file:
        frame_writer = csv.DictWriter(frame_file, fieldnames=[
            "video", "true_video_label", "frame_index", "frame_file", "video_anomaly_score",
            "frame_anomaly_score", "frame_object_anomaly_score", "predicted_is_anomalous_frame",
            "spatial_map",
        ])
        object_writer = csv.DictWriter(object_file, fieldnames=[
            "video", "frame_index", "frame_file", "object_index", "object_name",
            "object_anomaly_score", "mask_file",
        ])
        frame_writer.writeheader()
        object_writer.writeheader()
        for video_index, (video, true_label) in enumerate(ds.rows):
            video_path = Path(video)
            if not video_path.is_absolute():
                video_path = root / video_path
            frame_paths = sorted(
                path for path in video_path.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            cache_key = video_cache_key(video)
            mask_video_dir = segmentation_root / cache_key
            metadata_path = mask_video_dir / "objects.json"
            records = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else []
            records_by_frame = {}
            for record in records:
                records_by_frame.setdefault(int(record["frame"]), []).append(record)

            frame_indices = metrics["frame_indices"][video_index]
            object_scores = metrics["object_anomaly_scores"][video_index]
            for score_position, (frame_index, frame_score) in enumerate(zip(
                frame_indices, metrics["frame_anomaly_scores"][video_index]
            )):
                frame_index = int(frame_index)
                frame_path = frame_paths[frame_index]
                frame_records = records_by_frame.get(frame_index, [])
                masks = []
                mask_scores = []
                for record in frame_records:
                    object_index = int(record["object_index"])
                    if object_index >= len(object_scores[score_position]):
                        continue
                    score = float(object_scores[score_position][object_index])
                    mask_name = str(record["mask"])
                    mask_path = mask_video_dir / mask_name
                    if mask_path.is_file():
                        with Image.open(mask_path) as mask_image:
                            masks.append(np.asarray(mask_image.convert("L")))
                        mask_scores.append(score)
                    object_writer.writerow({
                        "video": video,
                        "frame_index": frame_index,
                        "frame_file": frame_path.name,
                        "object_index": object_index,
                        "object_name": record.get("object_name", ""),
                        "object_anomaly_score": score,
                        "mask_file": str(mask_path),
                    })

                with Image.open(frame_path) as frame_image:
                    output_size = frame_image.size
                spatial_map = compose_spatial_anomaly_map(masks, mask_scores, output_size)
                spatial_path = spatial_maps_root / cache_key / f"frame_{frame_index:06d}.png"
                spatial_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.round(spatial_map * 255).astype(np.uint8)).save(spatial_path)
                frame_writer.writerow({
                    "video": video,
                    "true_video_label": true_label,
                    "frame_index": frame_index,
                    "frame_file": frame_path.name,
                    "video_anomaly_score": metrics["video_anomaly_scores"][video_index],
                    "frame_anomaly_score": frame_score,
                    "frame_object_anomaly_score": metrics["frame_object_anomaly_scores"][video_index][score_position],
                    "predicted_is_anomalous_frame": int(frame_score >= args.anomaly_threshold),
                    "spatial_map": str(spatial_path),
                })
    print(
        f"accuracy={metrics['acc']:.6f} branch_accuracy={metrics['branch_acc']:.6f} "
        f"anomaly_accuracy={metrics['anomaly_acc']:.6f} loss={metrics['loss']:.6f} "
        f"predictions={predictions_path} frame_scores={frame_scores_path} "
        f"object_scores={object_scores_path} spatial_maps={spatial_maps_root}"
    )


if __name__ == "__main__":
    main()
