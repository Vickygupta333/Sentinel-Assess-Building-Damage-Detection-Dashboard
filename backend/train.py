"""
Training script for the real TwinTowerCNN (paper-faithful).

This is a skeleton, not a runnable-out-of-the-box trainer: you need real
labeled data first (UNOSAT/HDX damage assessments joined with satellite
imagery, as described in the paper and in the project README). Plug your
dataset into `PatchDataset` below and run:

    pip install torch torchvision
    python train.py --data_dir /path/to/patches --epochs 20

Expected data layout (one row per building patch):
    data_dir/
      labels.csv        # columns: patch_id,label  (label: 0=undamaged,1=damaged)
      pre/<patch_id>.png
      post/<patch_id>.png

Each patch should already be a 161x161 RGB crop (see pipeline.crop_patch).
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np


class PatchDataset:
    """Minimal Dataset wrapping pre/post patch pairs + binary labels.

    Kept dependency-light (plain PIL + numpy) so it's easy to adapt; wrap in
    torch.utils.data.Dataset once torch is installed.
    """

    def __init__(self, data_dir: str, augment: bool = True):
        self.data_dir = data_dir
        self.augment = augment
        self.rows = []
        with open(os.path.join(data_dir, "labels.csv")) as f:
            for row in csv.DictReader(f):
                self.rows.append((row["patch_id"], int(row["label"])))

    def __len__(self):
        return len(self.rows)

    def _load(self, patch_id: str, split: str):
        from PIL import Image

        path = os.path.join(self.data_dir, split, f"{patch_id}.png")
        return np.array(Image.open(path).convert("RGB"))

    def _augment(self, pre: np.ndarray, post: np.ndarray):
        # Random flips/rotations applied identically to both images so the
        # pre/post alignment is preserved — matches the paper's augmentation.
        if np.random.rand() < 0.5:
            pre, post = np.fliplr(pre), np.fliplr(post)
        if np.random.rand() < 0.5:
            pre, post = np.flipud(pre), np.flipud(post)
        k = np.random.randint(0, 4)
        pre, post = np.rot90(pre, k), np.rot90(post, k)
        return pre.copy(), post.copy()

    def __getitem__(self, idx):
        patch_id, label = self.rows[idx]
        pre = self._load(patch_id, "pre")
        post = self._load(patch_id, "post")
        if self.augment:
            pre, post = self._augment(pre, post)
        return pre, post, label


def train(data_dir: str, epochs: int, batch_size: int, lr: float, out_path: str):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    from model import build_twin_tower_cnn

    base_ds = PatchDataset(data_dir, augment=True)

    class TorchWrap(Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            pre, post, label = self.base[idx]
            pre_t = torch.from_numpy(pre.astype(np.float32) / 255.0).permute(2, 0, 1)
            post_t = torch.from_numpy(post.astype(np.float32) / 255.0).permute(2, 0, 1)
            return pre_t, post_t, torch.tensor(label, dtype=torch.float32)

    loader = DataLoader(TorchWrap(base_ds), batch_size=batch_size, shuffle=True)

    model = build_twin_tower_cnn()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for pre, post, label in loader:
            optimizer.zero_grad()
            pred = model(pre, post)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * pre.size(0)
        print(f"epoch {epoch + 1}/{epochs}  loss={total_loss / len(base_ds):.4f}")

    torch.save(model.state_dict(), out_path)
    print(f"saved weights to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="tts_model.pt")
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.out)
