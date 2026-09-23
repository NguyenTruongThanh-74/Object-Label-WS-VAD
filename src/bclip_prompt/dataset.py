from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageCSVClassificationDataset(Dataset):
    def __init__(self, csv_path: str, root: str = ".", image_size: int = 224, train: bool = True):
        self.root = Path(root)
        self.rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append((row["image"], int(row["label"])))

        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711)),
            ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rel, label = self.rows[idx]
        path = Path(rel)
        if not path.is_absolute():
            path = self.root / path
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
