"""A single, consistent logger configuration for every entry point.

Experimental code accumulates ``print`` calls that are impossible to silence
and impossible to redirect. Routing output through :mod:`logging` from the
first line of the project costs nothing and makes later instrumentation --
log files, per-rank filtering, verbosity flags -- a configuration change rather
than a rewrite.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

__all__ = ["get_logger", "configure_logging"]

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"
_configured = False


def configure_logging(level: int = logging.INFO, stream=None) -> None:
    """Install a stream handler on the root logger, exactly once."""
    global _configured
    if _configured:
        logging.getLogger().setLevel(level)
        return
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _configured = True


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Logger name; conventionally ``__name__`` of the calling module.
    level:
        Level applied to the root logger the first time logging is configured.
    """
    configure_logging(level)
    return logging.getLogger(name if name else "captioning")
