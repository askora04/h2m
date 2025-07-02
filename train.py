"""train.py
-----------
Example script that trains the reconstruction model.

Usage::

    python train.py --data_root /path/to/dataset --epochs 50 --batch_size 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import SkullFaceDataset
from model import ReconstructionModel


def parse_args():
    p = argparse.ArgumentParser(description="3D Face Reconstruction – Training")
    p.add_argument("--data_root", required=True, help="Folder containing *.png and *_vox.npz files")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default="checkpoints", help="Where to save weights")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resolution", type=int, default=64, help="Voxel resolution")
    return p.parse_args()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for binary volumes."""
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def train():
    args = parse_args()
    ds = SkullFaceDataset(args.data_root, resolution=args.resolution)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = ReconstructionModel(resolution=args.resolution).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for img, skull_gt, face_gt in loader:
            img = img.to(args.device)
            skull_gt = skull_gt.to(args.device)
            face_gt = face_gt.to(args.device)

            skull_pred, face_pred = model(img)
            loss_skull = dice_loss(skull_pred, skull_gt)
            loss_face = dice_loss(face_pred, face_gt)
            loss = loss_skull + loss_face

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs}  |  loss={avg_loss:.4f}")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_name = out_dir / f"recon_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, ckpt_name)
            print(f"Saved checkpoint {ckpt_name}")


if __name__ == "__main__":
    train()