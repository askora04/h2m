# 3D Face Reconstruction from 2D Skull Images

This project provides a minimal research‐grade pipeline that demonstrates how a convolutional neural network (CNN) can be trained to

1. Predict a 3-dimensional skull volume from a single 2-dimensional X-ray/CT projection (or photograph) of a skull.
2. Reconstruct the full 3-dimensional face (i.e. add soft-tissue "skin") from the predicted skull geometry.

Both steps are learned jointly in a single PyTorch model that has two cooperating heads.

> ⚠️  **Disclaimer**
> The code is for educational purposes only. Real forensic or medical reconstruction requires large proprietary datasets, domain knowledge and rigorous validation. The provided pipeline will not produce accurate anatomical reconstructions out-of-the-box.

---

## Project structure

```
.
├── dataset.py              # Lightweight PyTorch `Dataset` definitions
├── model.py                # CNN-based encoder-decoder for 3-D prediction
├── reconstruct_agent.py    # Inference wrapper ("agent")
├── train.py                # Training loop skeleton
├── requirements.txt        # Python dependencies
└── README.md
```

## Quick start (inference)

Assuming you have a trained checkpoint `weights.pt` and a skull image `sample.png`:

```bash
pip install -r requirements.txt
python reconstruct_agent.py --weights weights.pt --input sample.png --out_dir renders/
```

The script will

1. Predict a 3-D voxel grid for the skull.
2. Predict another voxel grid for the skin.
3. Save meshes (`*.ply`) and a quick-and-dirty Matplotlib render.

## Training

You need pairs of

* `skull_img`  : 2-D image of the skull (H×W×3)
* `skull_vox`  : Ground-truth 3-D binary voxel grid (D×H×W) for the skull
* `face_vox`   : Ground-truth 3-D binary voxel grid (D×H×W) for the face/skin

Put them into a folder structure such as

```
/your_dataset
    skull_000.png
    skull_000_vox.npz   # contains skull=..., face=...
    skull_001.png
    skull_001_vox.npz
    ...
```

Then launch training:

```bash
python train.py --data_root /your_dataset --epochs 50 --batch_size 4
```

Checkpoints will be saved under `checkpoints/`.

## Citation
If you build on this template for academic work, please cite it appropriately.