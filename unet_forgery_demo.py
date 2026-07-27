"""
Two-headed U-Net for copy-move forgery: detection + localization
================================================================

A minimal, self-contained demo of the task behind a scientific-image forgery
project: given an image, decide *whether* it has been tampered with (a patch
copied and pasted elsewhere) and *where*.

The network is a U-Net with a shared encoder and TWO output heads:

  * a classification head  -> "is this image forged?"  (one number per image)
  * a segmentation head    -> "which pixels are tampered?"  (a mask)

Both heads are trained together, sharing the same encoder features. Runs on
CPU in a couple of minutes.

Note on DINOv2
--------------
In the full project the shared encoder was replaced by features from DINOv2, a
large self-supervised vision transformer. DINOv2 is too large to download and
run inside a quick CPU demo, so this repository keeps a lightweight encoder and
focuses on the two-head mechanics. The place where DINOv2 features would slot
in is the encoder, marked in the code.

All images are synthetic. This is a mechanism demo, not a benchmark.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

IMG = 64
N_TRAIN = 600
N_TEST = 120


# ---------------------------------------------------------------------------
# 1. Synthetic data: half the images are forged, half are clean
# ---------------------------------------------------------------------------
# For a DETECTION head to mean anything, the dataset needs negatives too. So
# each image is either left clean (label 0, empty mask) or given a copy-move
# forgery (label 1, mask over the pasted region).
def make_image(forged):
    base = rng.normal(0, 1, (IMG, IMG))
    base = gaussian_filter(base, sigma=3.0)
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    img = base.copy()
    mask = np.zeros((IMG, IMG), dtype=np.float32)
    if forged:
        ps = rng.integers(12, 22)
        sy, sx = rng.integers(0, IMG - ps, size=2)     # source
        dy, dx = rng.integers(0, IMG - ps, size=2)     # destination
        patch = np.clip(base[sy:sy + ps, sx:sx + ps] + rng.normal(0, 0.05), 0, 1)
        img[dy:dy + ps, dx:dx + ps] = patch
        mask[dy:dy + ps, dx:dx + ps] = 1.0
    return img.astype(np.float32), mask, np.float32(forged)


def make_dataset(n):
    xs, ms, ls = [], [], []
    for i in range(n):
        img, mask, label = make_image(forged=(i % 2 == 0))   # half forged
        xs.append(img); ms.append(mask); ls.append(label)
    x = torch.tensor(np.stack(xs))[:, None]          # (n,1,H,W)
    m = torch.tensor(np.stack(ms))[:, None]          # (n,1,H,W)
    l = torch.tensor(np.stack(ls))[:, None]          # (n,1) image labels
    return x, m, l


train_x, train_m, train_l = make_dataset(N_TRAIN)
test_x, test_m, test_l = make_dataset(N_TEST)


# ---------------------------------------------------------------------------
# 2. Two-headed U-Net
# ---------------------------------------------------------------------------
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class TwoHeadUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        # --- Shared encoder ---
        # (In the full project these features came from DINOv2.)
        self.enc1 = conv_block(1, base)
        self.enc2 = conv_block(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base * 2, base * 4)

        # --- Head 1: segmentation (decoder back to full resolution) ---
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.seg_head = nn.Conv2d(base, 1, 1)

        # --- Head 2: classification (pool the bottleneck -> one logit) ---
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(base * 4, base * 2), nn.ReLU(inplace=True),
            nn.Linear(base * 2, 1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))           # shared features
        # segmentation branch
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        seg_logits = self.seg_head(d1)
        # detection branch
        cls_logits = self.cls_head(b)
        return cls_logits, seg_logits


model = TwoHeadUNet()


# ---------------------------------------------------------------------------
# 3. Losses (one per head) and metrics
# ---------------------------------------------------------------------------
bce = nn.BCEWithLogitsLoss()


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


@torch.no_grad()
def seg_iou(logits, target, thr=0.5):
    # Average IoU over FORGED images only (a clean image has no region to
    # localise, so an empty-vs-empty comparison would be undefined).
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum(dim=(1, 2, 3))
    union = ((p + target) > 0).float().sum(dim=(1, 2, 3))
    has_region = target.sum(dim=(1, 2, 3)) > 0
    iou = inter / (union + 1e-6)
    return iou[has_region].mean().item()


@torch.no_grad()
def cls_acc(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    return (pred == target).float().mean().item()


# ---------------------------------------------------------------------------
# 4. Train both heads together
# ---------------------------------------------------------------------------
# The total loss is simply the detection loss plus the segmentation loss; the
# shared encoder receives gradients from both, which is the whole idea of a
# multi-head network.
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 20
batch = 32

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(N_TRAIN)
    for i in range(0, N_TRAIN, batch):
        idx = perm[i:i + batch]
        xb, mb, lb = train_x[idx], train_m[idx], train_l[idx]
        cls_logits, seg_logits = model(xb)
        loss_cls = bce(cls_logits, lb)
        loss_seg = bce(seg_logits, mb) + dice_loss(seg_logits, mb)
        loss = loss_cls + loss_seg
        opt.zero_grad()
        loss.backward()
        opt.step()
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            c, s = model(test_x)
        print(f"epoch {epoch + 1:>2d}/{epochs}   "
              f"detection acc {cls_acc(c, test_l):.3f}   "
              f"segmentation IoU {seg_iou(s, test_m):.3f}")


# ---------------------------------------------------------------------------
# 5. Final evaluation + figure
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    cls_logits, seg_logits = model(test_x)
    acc = cls_acc(cls_logits, test_l)
    iou = seg_iou(seg_logits, test_m)
    cls_prob = torch.sigmoid(cls_logits)
    seg_prob = torch.sigmoid(seg_logits)
print(f"\nDetection head : accuracy {acc:.3f}")
print(f"Segmentation head : IoU {iou:.3f}")

# Show 2 forged + 2 clean test images so both heads are visible.
forged_idx = (test_l[:, 0] == 1).nonzero()[:2, 0].tolist()
clean_idx = (test_l[:, 0] == 0).nonzero()[:2, 0].tolist()
show = forged_idx + clean_idx

fig, ax = plt.subplots(3, 4, figsize=(12, 9))
for j, k in enumerate(show):
    p_forged = cls_prob[k, 0].item()
    ax[0, j].imshow(test_x[k, 0], cmap="gray")
    ax[0, j].set_title(f"input\ndetection: {p_forged:.0%} forged")
    ax[1, j].imshow(test_m[k, 0], cmap="magma", vmin=0, vmax=1)
    ax[1, j].set_title("true tampered region")
    ax[2, j].imshow(seg_prob[k, 0], cmap="magma", vmin=0, vmax=1)
    ax[2, j].set_title("segmentation head")
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"Two-headed U-Net: detection (is it forged?) + segmentation (where)   "
             f"[test acc {acc:.2f}, IoU {iou:.2f}]", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figure.png", dpi=120)
print("Saved figure.png")
