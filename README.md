# Two-headed U-Net for copy-move forgery: detection + localization

A minimal, self-contained demonstration of the task behind a scientific-image
forgery project. Given an image where a patch has been **copied and pasted**
elsewhere (a "copy-move" forgery), a U-Net with a **shared encoder and two
heads** answers both questions at once:

* a **detection** head — *is this image forged?* (one label per image)
* a **segmentation** head — *which pixels are tampered?* (a mask)

Both heads train together on the shared encoder features. It runs from scratch
on synthetic images on CPU in a couple of minutes.

![result](figure.png)

The figure shows held-out test images with the detection head's verdict
(top), the true tampered region (middle), and the segmentation head's
prediction (bottom) — including clean images, where the detector correctly
reports "not forged".

## What it does

- **Generates a balanced dataset.** Half the images are left clean; half get
  a copy-move forgery (a random square patch copied and pasted, leaving the
  faint boundary seam real pasting produces). Each image carries both a label
  (forged / clean) and a ground-truth mask.
- **Trains a two-headed U-Net.** A shared encoder feeds a segmentation decoder
  and a small classification head; the total loss is the detection loss plus
  the segmentation loss, so the shared encoder learns from both.
- **Evaluates** both heads: detection accuracy and segmentation IoU (measured
  on forged images, where there is a region to localise).

On synthetic data the detection head reaches ~**0.99 accuracy** and the
segmentation head ~**0.99 IoU**.

## Where DINOv2 fits

In the full project the **shared encoder** was replaced by features from
**DINOv2**, a large self-supervised vision transformer, which provides much
richer representations than a small convolutional encoder. DINOv2 is far too
large to download and run inside a quick CPU demo, so this repository keeps a
lightweight encoder and focuses on the two-head mechanics. The exact point
where DINOv2 features would slot in is the encoder, marked in the code.

## Run it

```bash
pip install -r requirements.txt
python unet_forgery_demo.py
```

CPU-only, a couple of minutes. It prints the training progress and test IoU
and regenerates `figure.png`.

## Scope and honesty

This is a **mechanism demo on synthetic data** — it shows the full forgery-
localization pipeline (data with masks, U-Net, segmentation loss, IoU) on an
easy-to-generate, easy-to-check version of the problem. It is not a benchmark,
and the synthetic detector keys on local paste artifacts rather than doing
full duplicate-region matching. The task and approach come from my master's
work on copy-move forgery detection in biomedical images; this repository
reimplements the segmentation core from scratch on toy data, with no project
data involved.

## References

- O. Ronneberger, P. Fischer, T. Brox, *U-Net: Convolutional Networks for
  Biomedical Image Segmentation*, MICCAI 2015.
- M. Oquab et al., *DINOv2: Learning Robust Visual Features without
  Supervision*, 2023.

## License

MIT — see [LICENSE](LICENSE).
