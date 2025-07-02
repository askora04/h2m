"""reconstruct_agent.py
----------------------
Command-line wrapper that loads the trained model and predicts 3-D skull and face
volumes from a single 2-D skull image. It then saves the voxel grids as `.npy`
files and optionally converts them to meshes (PLY) for visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from model import ReconstructionModel

try:
    import trimesh
except ImportError:
    trimesh = None


def parse_args():
    p = argparse.ArgumentParser(description="3D Face Reconstruction – Inference Agent")
    p.add_argument("--weights", required=True, help="Path to *.pt checkpoint")
    p.add_argument("--input", required=True, help="Input skull image (PNG/JPG)")
    p.add_argument("--out_dir", default="outputs", help="Folder to store predictions")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threshold", type=float, default=0.5, help="Occupancy threshold")
    p.add_argument("--resolution", type=int, default=64, help="Voxel resolution (must match training)")
    p.add_argument("--mesh", action="store_true", help="Export .ply meshes (requires trimesh)")
    return p.parse_args()


def voxel_to_mesh(vox: np.ndarray, out_path: Path):
    if trimesh is None:
        raise RuntimeError("trimesh not installed – `pip install trimesh` to enable mesh export")
    verts, faces, _, _ = trimesh.voxel.ops.matrix_to_marching_cubes(vox, pitch=1.0)
    mesh = trimesh.Trimesh(verts, faces)
    mesh.export(out_path.with_suffix(".ply"))


def main():
    args = parse_args()

    # Model
    model = ReconstructionModel(resolution=args.resolution)
    ckpt = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model = model.to(args.device)
    model.eval()

    # Preprocess image
    preprocess = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img = Image.open(args.input).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(args.device)

    # Predict
    with torch.no_grad():
        skull_occ, skin_occ = model.predict(img_t, threshold=args.threshold)
    skull_np = skull_occ.cpu().numpy()[0, 0]
    skin_np = skin_occ.cpu().numpy()[0, 0]

    # Save outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    np.save(out_dir / "skull.npy", skull_np)
    np.save(out_dir / "face.npy", skin_np)
    print(f"Saved voxel grids to {out_dir}")

    if args.mesh:
        print("Exporting meshes (MC) – this may take a moment…")
        voxel_to_mesh(skull_np, out_dir / "skull_mesh")
        voxel_to_mesh(skin_np, out_dir / "face_mesh")
        print("Meshes saved as .ply files.")


if __name__ == "__main__":
    main()