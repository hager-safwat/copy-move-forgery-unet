"""Tests for the synthetic forgery data, the two-headed U-Net and the metrics."""

import numpy as np
import pytest
import torch

from forgery import (
    TwoHeadUNet,
    cls_acc,
    dice_loss,
    false_positive_pixels,
    make_dataset,
    make_image,
    seg_iou,
    train,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_clean_image_has_empty_mask_and_label_zero():
    img, mask, label = make_image(False, np.random.default_rng(0))
    assert img.shape == mask.shape == (64, 64)
    assert mask.sum() == 0 and label == 0
    assert 0.0 <= img.min() and img.max() <= 1.0


def test_forged_image_mask_is_a_square_patch():
    _, mask, label = make_image(True, np.random.default_rng(0))
    assert label == 1
    rows, cols = np.nonzero(mask)
    h, w = np.ptp(rows) + 1, np.ptp(cols) + 1
    assert h == w and 12 <= h <= 21
    assert mask.sum() == h * w


def _source_found(img, mask):
    """True if the pasted block appears pixel-identical somewhere else."""
    rows, cols = np.nonzero(mask)
    ps = np.ptp(rows) + 1
    y0, x0 = rows.min(), cols.min()
    pasted = img[y0:y0 + ps, x0:x0 + ps]
    return any(
        np.array_equal(img[y:y + ps, x:x + ps], pasted)
        for y in range(64 - ps + 1) for x in range(64 - ps + 1)
        if (y, x) != (y0, x0)
    )


def test_exact_copy_reproduces_source_pixels():
    """With offset_std = 0 the pasted block is a true copy of another block
    of the image. It is not found only when the paste overwrote part of its
    own source (overlap), so we require it in most of 30 random images."""
    found = sum(
        _source_found(*make_image(True, np.random.default_rng(s), offset_std=0.0)[:2])
        for s in range(30)
    )
    assert found >= 20


def test_dataset_is_balanced_and_reproducible():
    x1, m1, l1 = make_dataset(10, np.random.default_rng(5))
    x2, _, _ = make_dataset(10, np.random.default_rng(5))
    assert x1.shape == m1.shape == (10, 1, 64, 64) and l1.shape == (10, 1)
    assert l1.sum() == 5
    assert torch.equal(x1, x2)


def test_model_output_shapes():
    model = TwoHeadUNet()
    c, s = model(torch.rand(3, 1, 64, 64))
    assert c.shape == (3, 1) and s.shape == (3, 1, 64, 64)


def test_parameter_count():
    """Pins the architecture size reported in the README."""
    assert sum(p.numel() for p in TwoHeadUNet().parameters()) == 119_506


def test_both_heads_send_gradient_to_shared_encoder():
    model = TwoHeadUNet()
    x = torch.rand(2, 1, 64, 64)
    for head in (0, 1):
        model.zero_grad()
        model(x)[head].sum().backward()
        assert model.enc1[0].weight.grad.abs().sum() > 0


def test_dice_loss_perfect_and_worst():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 2:6, 2:6] = 1
    perfect = (target * 2 - 1) * 50          # large logits of the right sign
    assert dice_loss(perfect, target).item() < 1e-3
    assert dice_loss(-perfect, target).item() > 0.9


def test_iou_ignores_clean_images_and_fp_counts_them():
    target = torch.zeros(2, 1, 8, 8)
    target[0, 0, :4, :4] = 1                 # image 0 forged, image 1 clean
    logits = torch.full((2, 1, 8, 8), -10.0)
    logits[0, 0, :4, :4] = 10.0              # perfect on the forged image
    logits[1, 0, :2, :3] = 10.0              # 6 false-positive px on clean
    assert seg_iou(logits, target) == pytest.approx(1.0, abs=1e-4)
    assert false_positive_pixels(logits, target) == pytest.approx(6.0)


def test_cls_acc():
    logits = torch.tensor([[3.0], [-3.0], [3.0]])
    labels = torch.tensor([[1.0], [0.0], [0.0]])
    assert cls_acc(logits, labels) == pytest.approx(2 / 3)


def test_training_reduces_loss():
    x, m, labels = make_dataset(32, np.random.default_rng(0))
    history = train(TwoHeadUNet(base=8), x, m, labels, epochs=3, batch=16, log_every=0)
    assert history[-1] < history[0]


def test_demo_smoke_run():
    import unet_forgery_demo

    unet_forgery_demo.N_TRAIN, unet_forgery_demo.N_TEST = 32, 16
    m = unet_forgery_demo.main(["--epochs", "1", "--no-figure"])
    assert 0.0 <= m["test_acc"] <= 1.0
