"""A baseline image captioning system, built for teaching.

The package is organised by responsibility rather than by training stage, so
that the same components are reused across the three stages of the tutorial:

    data/        corpus access, tokenisation, caption grounding
    models/      encoders, decoders, auxiliary heads
    training/    losses, optimisation loops, schedules
    inference/   decoding strategies
    evaluation/  metrics and reporting
    utils/       configuration, logging, reproducibility

Nothing in this package writes to disk or reads command-line arguments; that is
the responsibility of the entry points under ``scripts/``.
"""

__version__ = "0.1.0"
