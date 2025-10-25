"""Test configuration for path setup."""

from pathlib import Path
import sys


def pytest_sessionstart(session):  # type: ignore[override]
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
