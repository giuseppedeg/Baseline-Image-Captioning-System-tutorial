"""Image preprocessing.

Two decisions are made here, and both are frequent sources of silent error.

*Normalisation statistics must match the encoder's pre-training.* A backbone
pre-trained on ImageNet expects ImageNet channel statistics; a CLIP-derived
backbone expects CLIP's. Using the wrong pair does not raise an exception. It
degrades accuracy by a few points and is almost impossible to find later.

*Augmentation belongs to the training split only.* Validation and test images
are resized deterministically, so that a change in the metric reflects a change
in the model rather than a different random crop.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from torchvision import transforms

__all__ = ["IMAGENET_STATS", "CLIP_STATS", "build_transforms", "normalization_stats"]

#: (mean, std) per channel.
IMAGENET_STATS: Tuple[Sequence[float], Sequence[float]] = (
    (0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225),
)
CLIP_STATS: Tuple[Sequence[float], Sequence[float]] = (
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711),
)

_STATS = {"imagenet": IMAGENET_STATS, "clip": CLIP_STATS}


def normalization_stats(name: str) -> Tuple[Sequence[float], Sequence[float]]:
    try:
        return _STATS[name]
    except KeyError as exc:
        raise ValueError(f"unknown normalisation {name!r}; expected one of {sorted(_STATS)}") from exc


def build_transforms(
    image_size: int = 224,
    normalization: str = "imagenet",
    train: bool = False,
    scale: Tuple[float, float] = (0.7, 1.0),
) -> transforms.Compose:
    """Return the preprocessing pipeline for one split.

    Parameters
    ----------
    image_size:
        Side of the square crop fed to the encoder.
    normalization:
        ``imagenet`` or ``clip``; see the module docstring.
    train:
        When true, apply random resized cropping, horizontal flipping and a
        mild colour jitter. Horizontal flipping is safe for the architectural
        and object photography this tutorial targets; it would not be for text,
        diagrams or any subject with a canonical handedness.
    scale:
        Area range of the random crop, as a fraction of the source image.
        Aggressive cropping removes the global structure that period and style
        prediction depend on, so the default is conservative.
    """
    mean, std = normalization_stats(normalization)

    if train:
        steps = [
            transforms.RandomResizedCrop(image_size, scale=scale, ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        ]
    else:
        steps = [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
        ]

    steps += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(steps)
