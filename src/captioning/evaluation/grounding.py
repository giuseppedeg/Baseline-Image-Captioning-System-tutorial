"""Reference-free evaluation: does the caption describe *this* image?

N-gram metrics compare a caption to a sentence. CLIPScore compares it to the
image, which is the question actually being asked, and it does so without
needing a reference at all. On a corpus with one reference per image that
independence is worth a great deal.

The measure is a rescaled cosine similarity in CLIP's joint embedding space,
``2.5 * max(cos(v, t), 0)``, following Hessel et al. (2021). Two caveats belong
next to every number it produces: CLIP has its own biases and blind spots, and
a caption can be visually plausible and still assert something false -- which
is what :mod:`captioning.evaluation.factual` is for.

The dependency is optional. When it is unavailable this module reports that
fact and returns ``None`` rather than failing the evaluation run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from captioning.utils.logging import get_logger

__all__ = ["GroundingMetrics", "compute_grounding_metrics"]

logger = get_logger(__name__)

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


@dataclass
class GroundingMetrics:
    clip_score: float = 0.0
    #: Text-to-image retrieval over the evaluation set: is the caption closer
    #: to its own image than to the others? A weak captioner scores near chance
    #: even when its CLIPScore looks respectable, because its captions are
    #: generic rather than wrong.
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    n_samples: int = 0
    model: str = DEFAULT_CLIP_MODEL

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_grounding_metrics(
    images: Sequence[Union[str, Path]],
    captions: Sequence[str],
    model_name: str = DEFAULT_CLIP_MODEL,
    device: Optional[str] = None,
    batch_size: int = 32,
) -> Optional[GroundingMetrics]:
    """Embed images and captions with CLIP and score them.

    Returns ``None`` when ``transformers`` or the model weights are
    unavailable, after logging why.
    """
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        logger.warning(
            "CLIPScore skipped: install `transformers` to enable reference-free evaluation"
        )
        return None

    try:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("CLIPScore skipped: could not load %s (%s)", model_name, exc)
        return None

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    image_embeddings: List[torch.Tensor] = []
    text_embeddings: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(captions), batch_size):
            paths = images[start : start + batch_size]
            texts = list(captions[start : start + batch_size])
            batch_images = [Image.open(p).convert("RGB") for p in paths]

            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            image_embeddings.append(_normalize(model.get_image_features(**inputs)))

            inputs = processor(
                text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(device)
            text_embeddings.append(_normalize(model.get_text_features(**inputs)))

    image_matrix = torch.cat(image_embeddings)
    text_matrix = torch.cat(text_embeddings)

    paired = (image_matrix * text_matrix).sum(dim=-1).clamp(min=0.0)
    similarity = text_matrix @ image_matrix.T
    ranks = similarity.argsort(dim=-1, descending=True)
    truth = torch.arange(len(captions), device=ranks.device).unsqueeze(1)
    position = (ranks == truth).float().argmax(dim=-1)

    return GroundingMetrics(
        clip_score=float(2.5 * paired.mean()),
        recall_at_1=float((position < 1).float().mean()),
        recall_at_5=float((position < 5).float().mean()),
        recall_at_10=float((position < 10).float().mean()),
        n_samples=len(captions),
        model=model_name,
    )


def _normalize(tensor):
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-8)
