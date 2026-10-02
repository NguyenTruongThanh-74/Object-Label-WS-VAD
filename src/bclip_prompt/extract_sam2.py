from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from .bclip_adapter import BClipAdapter
from .dataset import IMAGE_SUFFIXES, video_cache_key
from .utils import load_json, load_yaml, resolve_device


DEFAULT_TEXT_TEMPLATES = [
    "a CCTV video frame containing {object}",
    "a surveillance camera view of {object}",
    "a video frame showing {object} during an incident",
]


@torch.no_grad()
def encode_texts(adapter: BClipAdapter, texts: list[str], device: torch.device) -> torch.Tensor:
    context_length = adapter.context_length
    token_ids = torch.zeros((len(texts), context_length), dtype=torch.long, device=device)
    eot_positions = []
    for row, text in enumerate(texts):
        words = adapter.token_ids(text)[: context_length - 2]
        sequence = [adapter.sot_id, *words, adapter.eot_id]
        token_ids[row, :len(sequence)] = torch.tensor(sequence, device=device)
        eot_positions.append(len(sequence) - 1)
    embeddings = adapter.embed_token_ids(token_ids)
    features = adapter.encode_soft_text(
        embeddings,
        torch.tensor(eot_positions, dtype=torch.long, device=device),
    )
    return torch.nn.functional.normalize(features, dim=-1)


def make_object_text_bank(adapter: BClipAdapter, classes: list[dict], cfg: dict, device: torch.device):
    templates = cfg.get("object_text_templates", DEFAULT_TEXT_TEMPLATES)
    vocabulary = cfg.get("object_vocabulary")
    if vocabulary is None:
        vocabulary = [
            {"name": str(category.get("object", category["name"])), "class_id": int(category["id"])}
            for category in classes
        ]
    if not templates or not vocabulary:
        raise ValueError("sam2.object_text_templates and sam2.object_vocabulary must not be empty")

    texts = []
    for item in vocabulary:
        object_name = str(item["name"])
        class_value = item.get("class_id")
        class_id = -1 if class_value is None else int(class_value)
        if class_id < -1 or class_id >= len(classes):
            raise ValueError(f"Object '{object_name}' has invalid class_id={class_id}")
        texts.extend(template.format(object=object_name) for template in templates)
    features = encode_texts(adapter, texts, device)
    bank = torch.nn.functional.normalize(features.reshape(len(vocabulary), len(templates), -1).mean(dim=1), dim=-1)
    class_ids = [int(item["class_id"]) for item in vocabulary]
    object_names = [str(item["name"]) for item in vocabulary]
    return bank, object_names, class_ids


def make_crop(rgb: np.ndarray, mask: np.ndarray, transform) -> torch.Tensor | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = rgb[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    crop[~crop_mask] = 0
    return transform(Image.fromarray(crop))


def mask_geometry(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros(5, dtype=np.float32)
    height, width = mask.shape
    x0, x1 = xs.min() / width, (xs.max() + 1) / width
    y0, y1 = ys.min() / height, (ys.max() + 1) / height
    return np.asarray([
        (x0 + x1) / 2.0,
        (y0 + y1) / 2.0,
        x1 - x0,
        y1 - y0,
        float(mask.mean()),
    ], dtype=np.float32)


@torch.no_grad()
def extract_video(
    video_path: Path,
    video_rel: str,
    mask_generator,
    adapter: BClipAdapter,
    object_text_bank: torch.Tensor,
    object_names: list[str],
    object_class_ids: list[int],
    cfg: dict,
    device: torch.device,
    cache_dir: Path,
    segmentation_dir: Path,
) -> None:
    scfg = cfg.get("sam2", {})
    dcfg = cfg["data"]
    max_objects = int(scfg.get("max_objects", 8))
    feature_dim = adapter.embed_dim
    frame_paths = sorted(
        path for path in video_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not frame_paths:
        raise ValueError(f"No image frames found in video directory: {video_path}")

    image_size = int(dcfg.get("image_size", 224))
    preprocess = transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    geometry_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST, antialias=False),
        transforms.CenterCrop(image_size),
    ])
    visual = np.zeros((len(frame_paths), max_objects, feature_dim), dtype=np.float32)
    text = np.zeros_like(visual)
    segmentation = np.zeros((len(frame_paths), max_objects, 5), dtype=np.float32)
    labels = np.full((len(frame_paths), max_objects), -1, dtype=np.int64)
    valid = np.zeros((len(frame_paths), max_objects), dtype=np.bool_)
    output_names = np.full((len(frame_paths), max_objects), "", dtype=f"U{max(map(len, object_names))}")
    records = []
    mask_video_dir = segmentation_dir / video_cache_key(video_rel)
    mask_video_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, frame_path in enumerate(tqdm(frame_paths, desc=video_rel, leave=False)):
        rgb = np.asarray(Image.open(frame_path).convert("RGB"))
        proposals = mask_generator.generate(rgb)
        proposals.sort(key=lambda item: float(item.get("predicted_iou", 0.0)), reverse=True)
        selected = []
        for proposal in proposals:
            mask = np.asarray(proposal["segmentation"], dtype=np.bool_)
            area_ratio = float(mask.mean())
            if area_ratio < float(scfg.get("min_mask_area_ratio", 0.001)):
                continue
            if area_ratio > float(scfg.get("max_mask_area_ratio", 0.8)):
                continue
            crop = make_crop(rgb, mask, preprocess)
            if crop is not None:
                selected.append((proposal, mask, crop))
            if len(selected) >= max_objects:
                break

        if not selected:
            continue
        crops = torch.stack([entry[2] for entry in selected]).to(device)
        crop_features = torch.nn.functional.normalize(adapter.encode_image_global(crops), dim=-1)
        similarities = crop_features @ object_text_bank.t()
        predicted_objects = similarities.argmax(dim=-1)
        for object_index, (proposal, mask, _) in enumerate(selected):
            object_index_in_vocab = int(predicted_objects[object_index].item())
            class_id = object_class_ids[object_index_in_vocab]
            object_name = object_names[object_index_in_vocab]
            visual[frame_index, object_index] = crop_features[object_index].cpu().numpy()
            text[frame_index, object_index] = object_text_bank[object_index_in_vocab].cpu().numpy()
            prepared_mask = geometry_transform(Image.fromarray(mask.astype(np.uint8) * 255))
            prepared_mask_array = np.asarray(prepared_mask) > 0
            segmentation[frame_index, object_index] = mask_geometry(prepared_mask_array)
            labels[frame_index, object_index] = class_id
            valid[frame_index, object_index] = True
            output_names[frame_index, object_index] = object_name
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_name = f"frame_{frame_index:06d}_object_{object_index:02d}_vocab_{object_index_in_vocab}.png"
            mask_image.save(mask_video_dir / mask_name)
            records.append({
                "frame": frame_index,
                "object_index": object_index,
                "object_name": object_name,
                "class_id": class_id,
                "mask_geometry": segmentation[frame_index, object_index].tolist(),
                "mask": mask_name,
                "similarity": float(similarities[object_index, object_index_in_vocab].item()),
            })

    np.savez_compressed(
        cache_dir / f"{video_cache_key(video_rel)}.npz",
        object_visual_features=visual,
        object_text_features=text,
        object_segmentation_features=segmentation,
        object_labels=labels,
        object_valid=valid,
        object_names=output_names,
    )
    with (mask_video_dir / "objects.json").open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Extract per-frame SAM2 masks and B-CLIP object proposals")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--csv", default=None, help="Video CSV; defaults to both train and validation CSVs")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = resolve_device(cfg.get("device", "cuda"))
    scfg = cfg.get("sam2", {})
    dcfg = cfg["data"]
    cache_dir = Path(scfg.get("cache_dir", "data/sam2_cache"))
    segmentation_dir = Path(scfg.get("segmentation_dir", "data/sam2_masks"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    if not scfg.get("checkpoint") or not scfg.get("model_cfg"):
        raise ValueError("Set sam2.checkpoint and sam2.model_cfg to your SAM2 checkpoint and model config")
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
    except ImportError as error:
        raise ImportError("Install the official Meta SAM2 package before extracting proposals") from error

    sam2_model = build_sam2(scfg["model_cfg"], scfg["checkpoint"], device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=int(scfg.get("points_per_side", 24)),
        pred_iou_thresh=float(scfg.get("pred_iou_thresh", 0.7)),
        stability_score_thresh=float(scfg.get("stability_score_thresh", 0.88)),
        min_mask_region_area=int(scfg.get("min_mask_region_area", 128)),
    )
    adapter = BClipAdapter(cfg["bclip"], device)
    classes = sorted(load_json(dcfg["classes_json"]), key=lambda item: int(item["id"]))
    if [int(item["id"]) for item in classes] != list(range(len(classes))):
        raise ValueError("Class ids in classes_json must be contiguous and start at zero")
    if int(dcfg.get("object_dim", adapter.embed_dim)) != adapter.embed_dim:
        raise ValueError(
            f"data.object_dim={dcfg['object_dim']} does not match β-CLIP embedding dimension {adapter.embed_dim}"
        )
    object_text_bank, object_names, object_class_ids = make_object_text_bank(adapter, classes, scfg, device)

    csv_paths = [Path(args.csv)] if args.csv else [Path(dcfg["train_csv"]), Path(dcfg["val_csv"])]
    videos = {}
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                videos[row["video"]] = row["video"]
    root = Path(dcfg.get("root", "."))
    for video_rel in videos:
        video_path = Path(video_rel)
        if not video_path.is_absolute():
            video_path = root / video_path
        extract_video(
            video_path,
            video_rel,
            mask_generator,
            adapter,
            object_text_bank,
            object_names,
            object_class_ids,
            cfg,
            device,
            cache_dir,
            segmentation_dir,
        )
    print(f"Wrote SAM2 proposal caches to {cache_dir} and masks to {segmentation_dir}")


if __name__ == "__main__":
    main()