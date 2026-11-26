"""U-Net with a shared encoder and two heads: detection and segmentation."""

import torch
from torch import nn


def conv_block(cin, cout):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class TwoHeadUNet(nn.Module):
    """Two-level U-Net.

    * Shared encoder: enc1 -> pool -> enc2 -> pool -> bottleneck.
      In the full project these features came from DINOv2; this is the
      place where a pretrained backbone would be plugged in.
    * Segmentation head: decoder with skip connections back to full
      resolution, one logit per pixel.
    * Detection head: global average pooling of the bottleneck, then a small
      MLP, one logit per image.

    Input size must be divisible by 4 (two pooling steps).
    """

    def __init__(self, base=16):
        super().__init__()
        self.enc1 = conv_block(1, base)
        self.enc2 = conv_block(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.seg_head = nn.Conv2d(base, 1, 1)

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(base * 4, base * 2), nn.ReLU(inplace=True),
            nn.Linear(base * 2, 1),
        )

    def forward(self, x):
        """x: (B, 1, H, W) -> (cls_logits (B, 1), seg_logits (B, 1, H, W))."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.cls_head(b), self.seg_head(d1)
