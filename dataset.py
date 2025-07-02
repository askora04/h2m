"""dataset.py
--------------
Minimal PyTorch `Dataset` helper for loading (image, skull_voxel, face_voxel)
triplets from a folder structure.

Skull voxel and face voxel ground-truth are expected as NumPy arrays saved in
`.npz` files with keys `skull` and `face` respectively.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class SkullFaceDataset(Dataset):
    """Dataset returning (image_tensor, skull_tensor, face_tensor)."""

    def __init__(self, root: str | Path, resolution: int = 64, transform: T.Compose | None = None):
        super().__init__()
        self.root = Path(root)
        self.resolution = resolution
        self.transform = transform or T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.img_paths = sorted([p for p in self.root.glob("*.png")])
        if not self.img_paths:
            raise RuntimeError(f"No .png images found in {self.root}")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.img_paths)

    def _vox_from_npz(self, npz_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        arrs = np.load(npz_path)
        skull = arrs["skull"][None]  # add channel dim
        face = arrs["face"][None]
        # Ensure correct resolution
        if skull.shape[-1] != self.resolution:
            raise ValueError(f"Voxel grids must be {self.resolution}^3, got {skull.shape}")
        skull_t = torch.from_numpy(skull).float()
        face_t = torch.from_numpy(face).float()
        return skull_t, face_t

    def __getitem__(self, idx: int):  # type: ignore[override]
        img_path = self.img_paths[idx]
        npz_path = img_path.with_suffix("_vox.npz")
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing voxel file {npz_path}")
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)
        skull_t, face_t = self._vox_from_npz(npz_path)
        return img_t, skull_t, face_t