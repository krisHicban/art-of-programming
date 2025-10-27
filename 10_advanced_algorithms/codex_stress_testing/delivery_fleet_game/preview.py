"""Entry point to launch the visualization prototype."""

from pathlib import Path
import sys


def bootstrap_src_path() -> None:
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> None:
    bootstrap_src_path()
    from ui.app import preview_latest_snapshot  # pylint: disable=import-error

    preview_latest_snapshot()


if __name__ == "__main__":
    main()
