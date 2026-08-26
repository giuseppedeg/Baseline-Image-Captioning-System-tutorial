"""Making the decoder's attention visible.

Cross-attention weights say which regions of the encoder grid the decoder read
while producing each token. They are not an explanation of the model's
reasoning -- attention is a soft read, not a justification -- but they are a
strong diagnostic. A model that attends to the sky while emitting *marble* has
learned the corpus prior, not the image, and no aggregate metric reveals that
as directly as the picture does.

The grid is coarse: 7x7 for a ResNet at 224 pixels, 14x14 for a ViT-B/16. The
maps are therefore upsampled for display, which makes them look smoother and
more confident than they are.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from captioning.data.transforms import normalization_stats

__all__ = ["denormalize", "attention_maps", "overlay_attention", "attention_figure"]


def denormalize(image: Tensor, normalization: str = "imagenet") -> Image.Image:
    """Invert the preprocessing of :mod:`captioning.data.transforms`."""
    mean, std = normalization_stats(normalization)
    tensor = image.detach().cpu().float()
    tensor = tensor * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
    array = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def attention_maps(attention: Tensor, grid: Tuple[int, int]) -> Tensor:
    """Reshape ``[T, N]`` attention weights into ``[T, H, W]`` maps."""
    height, width = grid
    steps, slots = attention.shape
    if height * width != slots:
        raise ValueError(
            f"attention has {slots} slots but the grid is {height}x{width}={height * width}; "
            "the encoder and the attention tensor disagree"
        )
    return attention.detach().cpu().float().reshape(steps, height, width)


def overlay_attention(
    image: Image.Image, weights: np.ndarray, alpha: float = 0.55, colour: str = "inferno"
) -> Image.Image:
    """Blend one attention map over an image.

    The map is normalised to its own range before display. This makes weak maps
    legible, and it also makes maps from different time steps incomparable in
    absolute terms -- read them for *where*, not for *how much*.
    """
    from matplotlib import cm

    span = float(weights.max() - weights.min())
    normalised = (weights - weights.min()) / span if span > 1e-8 else np.zeros_like(weights)

    heat = Image.fromarray((normalised * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC)
    coloured = cm.get_cmap(colour)(np.asarray(heat) / 255.0)[:, :, :3]
    coloured = Image.fromarray((coloured * 255).astype(np.uint8))
    return Image.blend(image.convert("RGB"), coloured, alpha=alpha)


def attention_figure(
    image: Tensor,
    tokens: Sequence[str],
    attention: Tensor,
    grid: Tuple[int, int],
    normalization: str = "imagenet",
    max_tokens: int = 12,
    columns: int = 5,
):
    """Return a matplotlib figure: the image, then one panel per token.

    Parameters
    ----------
    image:
        A single preprocessed image tensor, ``[3, H, W]``.
    tokens:
        The generated tokens, aligned with the first dimension of ``attention``.
    attention:
        ``[T, N]`` weights for the same example.
    """
    import matplotlib.pyplot as plt

    original = denormalize(image, normalization)
    maps = attention_maps(attention, grid).numpy()
    shown = min(len(tokens), maps.shape[0], max_tokens)

    panels = shown + 1
    rows = (panels + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(2.4 * columns, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    axes[0].imshow(original)
    axes[0].set_title("input", fontsize=9)
    axes[0].axis("off")

    for index in range(shown):
        axis = axes[index + 1]
        axis.imshow(overlay_attention(original, maps[index]))
        axis.set_title(tokens[index], fontsize=9)
        axis.axis("off")

    for axis in axes[panels:]:
        axis.axis("off")
    figure.tight_layout()
    return figure


def tokens_of(tokenizer, ids: Sequence[int]) -> List[str]:
    """Per-position strings for labelling attention panels.

    Decoding each identifier on its own, rather than the sequence as a whole,
    keeps the labels aligned with the attention rows even when the tokeniser
    merges subwords on a full decode.
    """
    specials = {tokenizer.pad_id, tokenizer.bos_id}
    return [tokenizer.decode([i], skip_special=False) for i in ids if i not in specials]
