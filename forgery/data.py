"""Synthetic copy-move forgery dataset: half the images forged, half clean."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

IMG = 64


def make_image(forged, rng, size=IMG, offset_std=0.05):
    """One grayscale image, its tampering mask and its image-level label.

    The background is smoothed Gaussian noise rescaled to [0, 1]. A forged
    image gets a random square patch (12 to 21 px) copied from one place and
    pasted at another, with a small brightness offset, as real pasting
    rarely matches exactly. The mask marks the pasted (destination) region.

    Returns
    -------
    img : np.ndarray (size, size), float32 in [0, 1]
    mask : np.ndarray (size, size), float32 in {0, 1}
    label : np.float32, 1.0 if forged else 0.0
    """
    base = gaussian_filter(rng.normal(0, 1, (size, size)), sigma=3.0)
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    img = base.copy()
    mask = np.zeros((size, size), dtype=np.float32)
    if forged:
        ps = rng.integers(12, 22)
        sy, sx = rng.integers(0, size - ps, size=2)     # source corner
        dy, dx = rng.integers(0, size - ps, size=2)     # destination corner
        # offset_std = 0 gives an exact copy (the generator is still called, so
        # the random sequence is the same either way).
        patch = np.clip(base[sy:sy + ps, sx:sx + ps] + rng.normal(0, offset_std), 0, 1)
        img[dy:dy + ps, dx:dx + ps] = patch
        mask[dy:dy + ps, dx:dx + ps] = 1.0
    return img.astype(np.float32), mask, np.float32(forged)


def make_dataset(n, rng, size=IMG, offset_std=0.05):
    """n images, alternating forged (even index) and clean (odd index).

    Returns tensors x (n, 1, H, W), masks (n, 1, H, W), labels (n, 1).
    """
    xs, ms, ls = [], [], []
    for i in range(n):
        img, mask, label = make_image(forged=(i % 2 == 0), rng=rng, size=size,
                                      offset_std=offset_std)
        xs.append(img)
        ms.append(mask)
        ls.append(label)
    x = torch.tensor(np.stack(xs))[:, None]
    m = torch.tensor(np.stack(ms))[:, None]
    labels = torch.tensor(np.stack(ls))[:, None]
    return x, m, labels
