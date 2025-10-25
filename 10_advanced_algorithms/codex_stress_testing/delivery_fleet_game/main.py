"""Entry point for the Delivery Fleet Management simulation."""

from pathlib import Path
import sys


def bootstrap_src_path() -> None:
    """Ensure the src directory is importable without installing a package."""
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> None:
    bootstrap_src_path()
    from core.engine import GameEngine  # pylint: disable=import-error

    engine = GameEngine()
    engine.run()


if __name__ == "__main__":
    main()
