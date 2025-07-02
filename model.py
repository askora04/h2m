"""model.py
---------------
Definitions of PyTorch modules that perform 3-D reconstruction from a 2-D skull image.

The core idea is a shared 2-D encoder that maps the input image to a latent vector.
Two separate decoders (3-D deconvolutional "heads") transform that latent space into
1. A voxelized binary occupancy grid of the skull (`head_skull`).
2. A second voxel grid representing the soft-tissue (skin) geometry (`head_skin`).

Both outputs have shape `(B, 1, D, H, W)` where each voxel stores the probability of
occupancy.

Typical resolution would be e.g. `64×64×64`. Higher resolutions are possible but
require more GPU memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock2D(nn.Module):
    """Simple 2-D residual block: conv → BN → ReLU → conv + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class Encoder2D(nn.Module):
    """Backbone that encodes the input image into a latent tensor."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        nf = [32, 64, 128, 256]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, nf[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(nf[0]),
            nn.ReLU(inplace=True),
        )
        # Four stages with residual blocks and downsampling
        layers = []
        c = nf[0]
        for n in nf[1:]:
            layers.append(nn.Conv2d(c, n, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(n))
            layers.append(nn.ReLU(inplace=True))
            layers.append(ResidualBlock2D(n))
            c = n
        self.backbone = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.stem(x)
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        latent = self.fc(x)
        return latent


class Decoder3D(nn.Module):
    """Generic 3-D decoder that upsamples a latent vector into a voxel grid."""

    def __init__(self, latent_dim: int = 256, out_channels: int = 1, resolution: int = 64):
        super().__init__()
        self.resolution = resolution
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4 * 4)

        # Series of 3-D transposed convolutions to reach desired resolution
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(True),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, out_channels, 1),  # Final conv to produce logits
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        b = latent.size(0)
        x = self.fc(latent).view(b, 512, 4, 4, 4)
        x = self.deconv(x)
        # Ensure correct output size (could differ by 1 due to rounding)
        if x.shape[-1] != self.resolution:
            x = F.interpolate(x, size=(self.resolution, self.resolution, self.resolution), mode="trilinear", align_corners=False)
        return x


class ReconstructionModel(nn.Module):
    """Full model comprised of a shared encoder and two decoders."""

    def __init__(self, latent_dim: int = 256, resolution: int = 64):
        super().__init__()
        self.encoder = Encoder2D(in_channels=3, latent_dim=latent_dim)
        self.decoder_skull = Decoder3D(latent_dim, out_channels=1, resolution=resolution)
        self.decoder_skin = Decoder3D(latent_dim, out_channels=1, resolution=resolution)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Returns skull_logits, skin_logits."""
        latent = self.encoder(img)
        skull_logits = self.decoder_skull(latent)
        skin_logits = self.decoder_skin(latent)
        return skull_logits, skin_logits

    def predict(self, img: torch.Tensor, threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        """Utility wrapper for inference that converts logits to binary occupancy."""
        self.eval()
        with torch.no_grad():
            skull_logits, skin_logits = self(img)
            skull_occ = (torch.sigmoid(skull_logits) > threshold).float()
            skin_occ = (torch.sigmoid(skin_logits) > threshold).float()
        return skull_occ, skin_occ