"""Typed configuration loaded from YAML.

Experiments are described by a YAML file rather than by command-line flags, so
that a run is fully reproducible from a single artefact that can be versioned
alongside the code. The YAML is parsed into dataclasses rather than kept as a
nested dictionary: the field names and types then document themselves, and a
typo in a key is reported at load time instead of silently taking a default.

Sections that a given stage does not yet consume are preserved verbatim in
``Config.extra``. This lets the same configuration file grow across the three
stages of the tutorial without breaking the entry points written for earlier
ones.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

import yaml

__all__ = [
    "ColumnMap",
    "Config",
    "ConfigError",
    "DataConfig",
    "DecoderConfig",
    "EncoderConfig",
    "InferenceConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TokenizerConfig",
    "TrainingConfig",
]


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or inconsistent."""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass
class ColumnMap:
    """Mapping from the column names of a corpus to the names used internally.

    Corpora arrive with whatever headers their authors chose. Rather than
    rewriting the data files -- an irreversible operation on someone else's
    data -- the mapping is declared in configuration and applied on load.
    """

    image_id: str = "ID"
    image_path: str = "Path"
    name: str = "name"
    century: str = "century"
    caption: str = "caption"

    #: Configuration key -> the name the column is known by inside the package.
    #: The two differ where the configuration key reads better qualified
    #: (``image_path``) than the internal one needs to be (``path``).
    CANONICAL = {
        "image_id": "id",
        "image_path": "path",
        "name": "name",
        "century": "century",
        "caption": "caption",
    }

    def as_dict(self) -> Dict[str, str]:
        """Configuration key -> column name in the user's CSV."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def canonical_mapping(self) -> Dict[str, str]:
        """Column name in the user's CSV -> canonical name used internally."""
        return {getattr(self, key): canonical for key, canonical in self.CANONICAL.items()}


@dataclass
class DataConfig:
    """Where the corpus lives and how it should be interpreted."""

    image_root: Path
    train_csv: Path
    val_csv: Optional[Path] = None
    test_csv: Optional[Path] = None
    columns: ColumnMap = field(default_factory=ColumnMap)

    #: Column supervising the decoder. See docs/01_problem_formulation.md.
    caption_field: str = "caption_grounded"

    image_size: int = 224
    #: ``imagenet`` or ``clip``; must match the encoder's pre-training.
    normalization: str = "imagenet"
    max_caption_length: int = 64

    #: Values in the century column that mean "unknown" rather than a century.
    century_unknown_values: List[int] = field(default_factory=lambda: [-1])

    #: Check at construction time that every referenced image file exists.
    verify_paths: bool = True

    def __post_init__(self) -> None:
        if self.normalization not in {"imagenet", "clip"}:
            raise ConfigError(
                f"data.normalization must be 'imagenet' or 'clip', got {self.normalization!r}"
            )
        if self.image_size <= 0:
            raise ConfigError("data.image_size must be positive")
        if self.max_caption_length < 3:
            # Two of the positions are always <bos> and <eos>.
            raise ConfigError("data.max_caption_length must be at least 3")


@dataclass
class TokenizerConfig:
    """Which tokenisation scheme to fit and where to persist it."""

    #: ``word`` for whitespace/punctuation tokens, ``bpe`` for subword units.
    kind: str = "word"
    vocab_size: int = 8000
    min_frequency: int = 2
    lowercase: bool = True
    path: Path = Path("artifacts/tokenizer")

    def __post_init__(self) -> None:
        if self.kind not in {"word", "bpe"}:
            raise ConfigError(f"tokenizer.kind must be 'word' or 'bpe', got {self.kind!r}")
        if self.vocab_size < 16:
            raise ConfigError("tokenizer.vocab_size is implausibly small")

    @property
    def artifact_path(self) -> Path:
        """Concrete file written by :mod:`captioning.data.tokenizer`."""
        suffix = "word.json" if self.kind == "word" else "bpe.json"
        return self.path / suffix


# ---------------------------------------------------------------------------
# Stage 1 sections
# ---------------------------------------------------------------------------


@dataclass
class EncoderConfig:
    """The frozen (or progressively unfrozen) visual backbone."""

    name: str = "resnet50"
    #: ``torchvision`` or ``timm``. The former has no extra dependency; the
    #: latter gives access to the vision transformers used from Stage 2 on.
    source: str = "torchvision"
    pretrained: bool = True
    #: Stage 1 trains the decoder only. Stage 2 revisits this.
    freeze: bool = True
    #: Number of trailing stages left trainable when ``freeze`` is false.
    trainable_stages: int = 0

    def __post_init__(self) -> None:
        if self.source not in {"torchvision", "timm"}:
            raise ConfigError(f"model.encoder.source must be 'torchvision' or 'timm', got {self.source!r}")


@dataclass
class DecoderConfig:
    """The caption decoder trained from scratch."""

    #: ``lstm`` reproduces Show, Attend and Tell; ``transformer`` is the
    #: architecture everything after 2017 is built on.
    kind: str = "transformer"
    d_model: int = 512
    dropout: float = 0.1
    #: Transformer only.
    n_heads: int = 8
    n_layers: int = 3
    ffn_dim: int = 2048
    #: LSTM only.
    hidden_dim: int = 512
    attention_dim: int = 512
    #: Tie the output projection to the embedding matrix. Saves parameters and
    #: usually helps on small corpora.
    tie_weights: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"lstm", "transformer"}:
            raise ConfigError(f"model.decoder.kind must be 'lstm' or 'transformer', got {self.kind!r}")
        if self.kind == "transformer" and self.d_model % self.n_heads:
            raise ConfigError(
                f"model.decoder.d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )


@dataclass
class ModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.98])
    #: Learning rate for encoder parameters once they are unfrozen. Fine-tuning
    #: a pre-trained backbone at the decoder's learning rate destroys it.
    encoder_lr: Optional[float] = None


@dataclass
class SchedulerConfig:
    warmup_steps: int = 500
    #: Floor of the cosine decay, as a fraction of the peak learning rate.
    min_lr_factor: float = 0.05


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    num_workers: int = 4
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    #: Mixed precision. Ignored on CPU.
    amp: bool = True
    output_dir: Path = Path("runs/stage1")
    #: Epochs without validation improvement before stopping. 0 disables it.
    early_stopping_patience: int = 5
    log_every: int = 25

    def __post_init__(self) -> None:
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ConfigError("training.label_smoothing must lie in [0, 1)")


@dataclass
class InferenceConfig:
    """How captions are produced at evaluation time."""

    strategy: str = "beam"
    beam_size: int = 3
    #: Exponent of the length normalisation applied to beam scores. Without it
    #: beam search systematically prefers short captions.
    length_penalty: float = 0.7
    max_new_tokens: int = 48
    temperature: float = 1.0
    top_p: float = 0.9

    def __post_init__(self) -> None:
        if self.strategy not in {"greedy", "beam", "nucleus"}:
            raise ConfigError(
                f"inference.strategy must be 'greedy', 'beam' or 'nucleus', got {self.strategy!r}"
            )
        if self.strategy == "beam" and self.beam_size < 1:
            raise ConfigError("inference.beam_size must be at least 1")


# ---------------------------------------------------------------------------
# Root object
# ---------------------------------------------------------------------------

_KNOWN_SECTIONS = {"data", "tokenizer", "model", "training", "inference", "seed"}


@dataclass
class Config:
    """The parsed contents of a configuration file."""

    data: DataConfig
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    seed: int = 1234
    #: Sections not consumed by the current stage, preserved unchanged.
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} must contain a YAML mapping at the top level")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        if "data" not in raw:
            raise ConfigError("configuration must contain a 'data' section")
        return cls(
            data=build(DataConfig, raw["data"], path="data"),
            tokenizer=build(TokenizerConfig, raw.get("tokenizer", {}), path="tokenizer"),
            model=build(ModelConfig, raw.get("model", {}), path="model"),
            training=build(TrainingConfig, raw.get("training", {}), path="training"),
            inference=build(InferenceConfig, raw.get("inference", {}), path="inference"),
            seed=int(raw.get("seed", 1234)),
            extra={k: v for k, v in raw.items() if k not in _KNOWN_SECTIONS},
        )

    def section(self, name: str) -> Dict[str, Any]:
        """Return a not-yet-typed section, or raise a helpful error.

        Later stages of the tutorial replace these calls with proper
        dataclasses; until then this keeps the failure mode explicit.
        """
        if name not in self.extra:
            raise ConfigError(
                f"configuration section {name!r} is required by this entry point "
                f"but is absent from the file"
            )
        return self.extra[name]

    def as_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        extra = out.pop("extra")
        out.update(extra)
        return _stringify_paths(out)


# ---------------------------------------------------------------------------
# Dictionary -> dataclass coercion
# ---------------------------------------------------------------------------


def build(cls: type, data: Any, path: str = "") -> Any:
    """Instantiate the dataclass ``cls`` from a mapping, coercing field types.

    Unknown keys raise :class:`ConfigError` rather than being ignored. A
    silently ignored key is one of the more expensive kinds of bug in
    experimental code: the run completes, the numbers look reasonable, and the
    setting one believed was applied never was.
    """
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path or cls.__name__} must be a mapping, got {type(data).__name__}")

    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in section {path or cls.__name__!r}: {', '.join(unknown)}. "
            f"Valid keys are: {', '.join(sorted(known))}"
        )

    kwargs = {}
    for key, value in data.items():
        child = f"{path}.{key}" if path else key
        kwargs[key] = _coerce(value, hints[key], child)
    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except TypeError as exc:  # missing required field
        raise ConfigError(f"invalid section {path or cls.__name__!r}: {exc}") from exc


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    origin = get_origin(annotation)

    if origin is Union:  # includes Optional[T]
        args = [a for a in get_args(annotation) if a is not type(None)]  # noqa: E721
        if value is None:
            return None
        return _coerce(value, args[0], path)

    if is_dataclass(annotation):
        return build(annotation, value, path)

    if annotation is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"{path} must be a path, got {type(value).__name__}")
        return Path(value)

    if origin in (list, List):
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a list, got {type(value).__name__}")
        args = get_args(annotation)
        item_type = args[0] if args else Any
        return [_coerce(v, item_type, f"{path}[{i}]") for i, v in enumerate(value)]

    return value


def _stringify_paths(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_paths(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
