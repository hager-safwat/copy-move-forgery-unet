"""
Two-headed U-Net for copy-move forgery: detection + localisation
================================================================

Trains a U-Net with a shared encoder and two heads on synthetic copy-move
forgeries: a detection head ("is this image forged?") and a segmentation
head ("which pixels are tampered?"). Evaluates both on held-out images,
runs one stress test, and draws the figure.

All images are synthetic. This is a mechanism demo, not a benchmark.

Usage
-----
    python unet_forgery_demo.py                         # full run, writes figure.png
    python unet_forgery_demo.py --epochs 1 --no-figure  # quick smoke run
"""

import argparse

import numpy as np
import torch

from forgery import (
    TwoHeadUNet,
    cls_acc,
    false_positive_pixels,
    make_dataset,
    seg_iou,
    train,
)

SEED = 0
N_TRAIN = 600
N_TEST = 120


@torch.no_grad()
def evaluate(model, x, m, labels):
    model.eval()
    c, s = model(x)
    return {
        "acc": cls_acc(c, labels),
        "iou": seg_iou(s, m),
        "fp_pixels_clean": false_positive_pixels(s, m),
        "cls_prob": torch.sigmoid(c),
        "seg_prob": torch.sigmoid(s),
    }


def run_experiment(epochs=20, seed=SEED):
    """Order matters for exact reproducibility: one numpy generator makes
    the train then test sets, one torch seed controls init and shuffling."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    train_x, train_m, train_l = make_dataset(N_TRAIN, rng)
    test_x, test_m, test_l = make_dataset(N_TEST, rng)

    model = TwoHeadUNet()
    n_params = sum(p.numel() for p in model.parameters())
    train(model, train_x, train_m, train_l, epochs=epochs,
          eval_data=(test_x, test_m, test_l), log_every=5)
    test = evaluate(model, test_x, test_m, test_l)

    # Stress test: forgeries pasted with NO brightness offset (an exact copy).
    # Generated with a separate generator so the main run is unaffected.
    sx, sm, sl = make_dataset(N_TEST, np.random.default_rng(seed + 1), offset_std=0.0)
    stress = evaluate(model, sx, sm, sl)

    metrics = {
        "n_params": n_params,
        "test_acc": test["acc"],
        "test_iou": test["iou"],
        "test_fp_pixels_clean": test["fp_pixels_clean"],
        "exact_copy_acc": stress["acc"],
        "exact_copy_iou": stress["iou"],
    }
    return metrics, (test_x, test_m, test_l), test


def print_metrics(m):
    print(f"\ntrainable parameters                       {m['n_params']:,}")
    print(f"held-out test, detection accuracy          {m['test_acc']:.3f}")
    print(f"held-out test, segmentation IoU (forged)   {m['test_iou']:.3f}")
    print(f"held-out test, false-positive px (clean)   {m['test_fp_pixels_clean']:.1f}")
    print("\nStress test: exact copies, no brightness offset")
    print(f"  detection accuracy                       {m['exact_copy_acc']:.3f}")
    print(f"  segmentation IoU (forged)                {m['exact_copy_iou']:.3f}")


def save_figure(test_data, test, metrics, path="figure.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test_x, test_m, test_l = test_data
    forged_idx = (test_l[:, 0] == 1).nonzero()[:2, 0].tolist()
    clean_idx = (test_l[:, 0] == 0).nonzero()[:2, 0].tolist()

    fig, ax = plt.subplots(3, 4, figsize=(12, 9))
    for j, k in enumerate(forged_idx + clean_idx):
        ax[0, j].imshow(test_x[k, 0], cmap="gray")
        ax[0, j].set_title(f"input\ndetection: {test['cls_prob'][k, 0].item():.0%} forged")
        ax[1, j].imshow(test_m[k, 0], cmap="magma", vmin=0, vmax=1)
        ax[1, j].set_title("true tampered region")
        ax[2, j].imshow(test["seg_prob"][k, 0], cmap="magma", vmin=0, vmax=1)
        ax[2, j].set_title("segmentation head")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Two-headed U-Net: detection (is it forged?) + segmentation (where)   "
                 f"[test acc {metrics['test_acc']:.2f}, IoU {metrics['test_iou']:.2f}]",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=120)
    print(f"\nSaved {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a two-headed U-Net on synthetic forgeries.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-figure", action="store_true", help="skip figure.png")
    args = parser.parse_args(argv)

    metrics, test_data, test = run_experiment(epochs=args.epochs, seed=args.seed)
    print_metrics(metrics)
    if not args.no_figure:
        save_figure(test_data, test, metrics)
    return metrics


if __name__ == "__main__":
    main()
