"""Make ``src/`` importable when the package has not been installed.

Running ``pip install -e .`` is the recommended setup and makes this module
redundant. It exists so that a reader who has just cloned the repository can
execute the scripts immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
