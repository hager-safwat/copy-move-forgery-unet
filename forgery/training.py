"""Losses, metrics and the joint training loop for the two heads."""

import torch
from torch import nn

bce = nn.BCEWithLogitsLoss()


def dice_loss(logits, target, eps=1.0):
    """Soft Dice loss per image, averaged over the batch.

    eps smooths the ratio so that a clean image (empty mask) predicted as
    empty gets a loss close to 0 instead of 0/0.
    """
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


@torch.no_grad()
def seg_iou(logits, target, thr=0.5):
    """Mean IoU of the thresholded mask, over FORGED images only.

    A clean image has no region to localise, so empty-vs-empty IoU is
    undefined and those images are excluded.
    """
    p = (torch.sigmoid(logits) > thr).float()
    inter = (p * target).sum(dim=(1, 2, 3))
    union = ((p + target) > 0).float().sum(dim=(1, 2, 3))
    has_region = target.sum(dim=(1, 2, 3)) > 0
    return (inter / (union + 1e-6))[has_region].mean().item()


@torch.no_grad()
def false_positive_pixels(logits, target, thr=0.5):
    """Average number of pixels predicted as tampered on CLEAN images.

    Complements seg_iou, which ignores clean images.
    """
    p = (torch.sigmoid(logits) > thr).float()
    clean = target.sum(dim=(1, 2, 3)) == 0
    return p[clean].sum(dim=(1, 2, 3)).mean().item()


@torch.no_grad()
def cls_acc(logits, target):
    """Image-level detection accuracy at threshold 0.5."""
    return ((torch.sigmoid(logits) > 0.5).float() == target).float().mean().item()


def joint_loss(cls_logits, seg_logits, labels, masks):
    """Detection BCE + segmentation (BCE + Dice). The shared encoder gets
    gradients from both heads."""
    return bce(cls_logits, labels) + bce(seg_logits, masks) + dice_loss(seg_logits, masks)


def train(model, train_x, train_m, train_l, epochs=20, batch=32, lr=1e-3,
          eval_data=None, log_every=5):
    """Mini-batch Adam on the joint loss. Returns per-epoch mean losses.

    If ``eval_data = (x, m, l)`` is given, prints accuracy and IoU on it
    every ``log_every`` epochs (monitoring only, nothing is selected on it).
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_x.shape[0]
    history = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_losses = []
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            cls_logits, seg_logits = model(train_x[idx])
            loss = joint_loss(cls_logits, seg_logits, train_l[idx], train_m[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        history.append(sum(epoch_losses) / len(epoch_losses))
        if eval_data is not None and log_every and (epoch + 1) % log_every == 0:
            model.eval()
            with torch.no_grad():
                c, s = model(eval_data[0])
            print(f"epoch {epoch + 1:>2d}/{epochs}   "
                  f"detection acc {cls_acc(c, eval_data[2]):.3f}   "
                  f"segmentation IoU {seg_iou(s, eval_data[1]):.3f}")
    return history
