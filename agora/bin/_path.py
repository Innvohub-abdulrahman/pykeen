"""Put ``src`` on the path so the reference scripts run from a clean checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> None:
    src = Path(__file__).resolve().parent.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
