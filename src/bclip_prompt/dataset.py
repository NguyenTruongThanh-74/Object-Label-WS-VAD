from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def video_cache_key(video_path: str) -> str:
    normalized = Path(video_path).as_posix().rstrip("/")
    safe_name = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in Path(normalized).name
    )
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{safe_name}_{digest}"


class VideoCSVClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root: str = ".",
        image_size: int = 224,
        frames_per_video: int | None = 8,
        cache_dir: str | None = None,
        max_objects: int = 8,
        object_dim: int = 512,
        require_cache: bool = True,
        train: bool = True,
    ):
        self.root = Path(root)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.frames_per_video = int(frames_per_video) if frames_per_video is not None else None
        self.max_objects = int(max_objects)
        self.object_dim = int(object_dim)
        self.require_cache = require_cache
        self.train = train
        if (self.frames_per_video is not None and self.frames_per_video < 1) or self.max_objects < 1 or self.object_dim < 1:
            raise ValueError("frames_per_video, max_objects, and object_dim must be positive")

        self.rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append((row["video"], int(row["label"])))

        self.transform = transforms.Compose([
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.rows)

    def _sample_indices(self, count: int) -> np.ndarray:
        if self.frames_per_video is None:
            return np.arange(count)
        if count <= self.frames_per_video:
            return np.arange(self.frames_per_video) % count

        edges = np.linspace(0, count, self.frames_per_video + 1)
        if self.train:
            starts = np.floor(edges[:-1]).astype(int)
            ends = np.maximum(starts + 1, np.floor(edges[1:]).astype(int))
            return np.asarray([
                np.random.randint(start, min(end, count))
                for start, end in zip(starts, ends)
            ])
        return np.linspace(0, count - 1, self.frames_per_video).astype(int)

    def _load_proposals(self, video_rel: str, frame_indices: np.ndarray):
        sample_count = len(frame_indices)
        shape = (sample_count, self.max_objects, self.object_dim)
        visual = np.zeros(shape, dtype=np.float32)
        text = np.zeros(shape, dtype=np.float32)
        segmentation = np.zeros((sample_count, self.max_objects, 5), dtype=np.float32)
        labels = np.full((sample_count, self.max_objects), -1, dtype=np.int64)
        valid = np.zeros((sample_count, self.max_objects), dtype=np.bool_)
        if self.cache_dir is None:
            if self.require_cache:
                raise ValueError("data.sam2_cache_dir is required when SAM2 proposals are enabled")
            return (
                torch.from_numpy(visual), torch.from_numpy(text), torch.from_numpy(segmentation),
                torch.from_numpy(labels), torch.from_numpy(valid),
            )

        cache_path = self.cache_dir / f"{video_cache_key(video_rel)}.npz"
        if not cache_path.exists():
            if self.require_cache:
                raise FileNotFoundError(
                    f"SAM2 cache not found for video '{video_rel}': {cache_path}. "
                    "Run `python -m bclip_prompt.extract_sam2` first."
                )
            return (
                torch.from_numpy(visual), torch.from_numpy(text), torch.from_numpy(segmentation),
                torch.from_numpy(labels), torch.from_numpy(valid),
            )

        with np.load(cache_path) as cache:
            cached_visual = cache["object_visual_features"]
            cached_text = cache["object_text_features"]
            cached_segmentation = cache["object_segmentation_features"]
            cached_labels = cache["object_labels"]
            cached_valid = cache["object_valid"]
            if cached_visual.shape != cached_text.shape or cached_visual.ndim != 3:
                raise ValueError(f"Invalid object feature shapes in {cache_path}")
            if cached_labels.shape != cached_visual.shape[:2] or cached_valid.shape != cached_visual.shape[:2]:
                raise ValueError(f"Invalid object label or validity-mask shapes in {cache_path}")
            if cached_segmentation.shape != (*cached_visual.shape[:2], 5):
                raise ValueError(f"Invalid segmentation geometry shape in {cache_path}")
            if cached_visual.shape[-1] != self.object_dim:
                raise ValueError(
                    f"SAM2 cache dimension {cached_visual.shape[-1]} does not match "
                    f"data.object_dim={self.object_dim}"
                )
            for output_index, source_index in enumerate(frame_indices):
                if source_index >= cached_visual.shape[0]:
                    raise ValueError(f"SAM2 cache has fewer frames than video '{video_rel}'")
                count = min(self.max_objects, cached_visual.shape[1])
                visual[output_index, :count] = cached_visual[source_index, :count]
                text[output_index, :count] = cached_text[source_index, :count]
                segmentation[output_index, :count] = cached_segmentation[source_index, :count]
                labels[output_index, :count] = cached_labels[source_index, :count]
                valid[output_index, :count] = cached_valid[source_index, :count]

            return (
                torch.from_numpy(visual), torch.from_numpy(text), torch.from_numpy(segmentation),
                torch.from_numpy(labels), torch.from_numpy(valid),
            )

    def __getitem__(self, idx):
        video_rel, label = self.rows[idx]
        video_path = Path(video_rel)
        if not video_path.is_absolute():
            video_path = self.root / video_path
        if not video_path.is_dir():
            raise NotADirectoryError(f"Video sample must be a frame directory: {video_path}")
        frame_paths = sorted(
            path for path in video_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not frame_paths:
            raise ValueError(f"No image frames found in video directory: {video_path}")

        frame_indices = self._sample_indices(len(frame_paths))
        frame_tensors = []
        for frame_index in frame_indices:
            with Image.open(frame_paths[int(frame_index)]) as image:
                frame_tensors.append(self.transform(image.convert("RGB")))
        frames = torch.stack(frame_tensors)
        visual, text, segmentation, object_labels, valid = self._load_proposals(video_rel, frame_indices)
        return frames, label, visual, text, segmentation, object_labels, valid, torch.from_numpy(frame_indices.copy())
