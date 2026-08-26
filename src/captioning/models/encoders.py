"""Visual encoders.

The encoder turns an image into a sequence of feature vectors. Two properties
of that sequence matter downstream and neither is negotiable:

*It is a sequence, not a single vector.* A decoder that receives one pooled
vector must compress the whole image into it before generating a single word.
A decoder that receives the spatial grid can attend to different regions at
different time steps, which is both better and inspectable -- the attention
maps in :mod:`captioning.inference.attention` are only meaningful because the
encoder preserves spatial structure.

*It is frozen in Stage 1.* The backbone was pre-trained on far more images
than this corpus contains. Updating it with gradients from a randomly
initialised decoder destroys those features in the first few hundred steps,
long before the decoder produces a useful signal. Stage 2 unfreezes it, in the
right order and at the right learning rate.

Freezing correctly requires more than ``requires_grad_(False)``. A batch
normalisation layer in training mode keeps updating its running statistics
regardless of gradients, so a "frozen" ResNet silently drifts across epochs.
:meth:`VisualEncoder.train` therefore keeps frozen submodules in evaluation
mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor, nn

from captioning.utils.config import EncoderConfig

__all__ = ["EncoderOutput", "VisualEncoder", "ResNetEncoder", "TimmEncoder", "build_encoder"]


@dataclass
class EncoderOutput:
    """Features extracted from a batch of images."""

    #: ``[B, N, D]`` -- one vector per spatial location or patch.
    tokens: Tensor
    #: ``[B, D]`` -- mean over locations, used to initialise recurrent decoders.
    pooled: Tensor
    #: ``(height, width)`` of the feature grid, for reshaping attention maps.
    grid: Tuple[int, int]


class VisualEncoder(nn.Module):
    """Base class fixing the interface and the freezing semantics."""

    #: Dimension of each feature vector.
    output_dim: int = 0

    def forward(self, images: Tensor) -> EncoderOutput:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- parameter groups --------------------------------------------------

    @property
    def stages(self) -> List[nn.Module]:
        """Blocks of the backbone, ordered from earliest to latest.

        Progressive unfreezing in Stage 2 walks this list from the end: late
        blocks encode task-specific structure and adapt usefully, early blocks
        encode edges and colours and rarely need to.
        """
        raise NotImplementedError

    def freeze(self) -> "VisualEncoder":
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def set_trainable_stages(self, n: int) -> "VisualEncoder":
        """Leave the last ``n`` stages trainable and freeze everything else."""
        self.freeze()
        if n > 0:
            for stage in self.stages[-n:]:
                for parameter in stage.parameters():
                    parameter.requires_grad_(True)
        return self

    @property
    def is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.parameters())

    def train(self, mode: bool = True) -> "VisualEncoder":
        """Keep frozen submodules in evaluation mode.

        Without this, batch-normalisation running statistics keep updating in a
        backbone that receives no gradients -- a frozen encoder whose outputs
        nevertheless change from epoch to epoch.
        """
        super().train(mode)
        if not mode:
            return self
        for module in self.modules():
            if module is self:
                continue
            params = list(module.parameters(recurse=False))
            if params and not any(p.requires_grad for p in params):
                module.eval()
            elif isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
                p.requires_grad for p in module.parameters(recurse=False)
            ):
                module.eval()
        return self


# ---------------------------------------------------------------------------
# torchvision backbones
# ---------------------------------------------------------------------------

_RESNET_DIMS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048, "resnet152": 2048}


class ResNetEncoder(VisualEncoder):
    """A torchvision ResNet truncated before its classification head.

    A 224x224 input yields a 7x7 grid of feature vectors: 49 tokens, which is a
    convenient size for cross-attention and small enough to visualise.
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = True) -> None:
        super().__init__()
        import torchvision.models as tvm

        if name not in _RESNET_DIMS:
            raise ValueError(f"unsupported ResNet {name!r}; expected one of {sorted(_RESNET_DIMS)}")
        weights = "DEFAULT" if pretrained else None
        backbone = getattr(tvm, name)(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.blocks = nn.ModuleList([backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4])
        self.output_dim = _RESNET_DIMS[name]
        self.name = name

    @property
    def stages(self) -> List[nn.Module]:
        return [self.stem, *self.blocks]

    def forward(self, images: Tensor) -> EncoderOutput:
        x = self.stem(images)
        for block in self.blocks:
            x = block(x)
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()  # [B, H*W, C]
        return EncoderOutput(tokens=tokens, pooled=tokens.mean(dim=1), grid=(height, width))


# ---------------------------------------------------------------------------
# timm backbones
# ---------------------------------------------------------------------------


class TimmEncoder(VisualEncoder):
    """Any ``timm`` backbone, used through ``forward_features``.

    Handles both output conventions: vision transformers return
    ``[B, prefix + N, D]`` and the prefix (class and register tokens) is
    dropped; convolutional backbones return ``[B, D, H, W]`` and are flattened.
    """

    def __init__(self, name: str = "vit_base_patch16_224", pretrained: bool = True) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "timm is required for this encoder:\n    pip install timm\n"
                "Alternatively set model.encoder.source to 'torchvision'."
            ) from exc

        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.n_prefix = int(getattr(self.backbone, "num_prefix_tokens", 0))
        self.output_dim = int(self.backbone.num_features)
        self.name = name

    @property
    def stages(self) -> List[nn.Module]:
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is not None:
            return list(blocks)
        stages = getattr(self.backbone, "stages", None)
        if stages is not None:
            return list(stages)
        return [self.backbone]

    def forward(self, images: Tensor) -> EncoderOutput:
        features = self.backbone.forward_features(images)
        if features.dim() == 4:  # [B, D, H, W]
            _, _, height, width = features.shape
            tokens = features.flatten(2).transpose(1, 2).contiguous()
        elif features.dim() == 3:  # [B, prefix + N, D]
            tokens = features[:, self.n_prefix :, :].contiguous()
            side = int(math.sqrt(tokens.shape[1]))
            if side * side != tokens.shape[1]:
                raise RuntimeError(
                    f"{self.name} produced {tokens.shape[1]} patch tokens, which is not a square "
                    "grid; attention maps cannot be reshaped for this backbone"
                )
            height = width = side
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"unexpected feature shape {tuple(features.shape)} from {self.name}")
        return EncoderOutput(tokens=tokens, pooled=tokens.mean(dim=1), grid=(height, width))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_encoder(config: EncoderConfig) -> VisualEncoder:
    """Construct the encoder described by the configuration and apply its
    freezing policy."""
    if config.source == "torchvision":
        encoder: VisualEncoder = ResNetEncoder(config.name, config.pretrained)
    else:
        encoder = TimmEncoder(config.name, config.pretrained)

    if config.freeze:
        encoder.freeze()
    else:
        encoder.set_trainable_stages(config.trainable_stages)
    return encoder
