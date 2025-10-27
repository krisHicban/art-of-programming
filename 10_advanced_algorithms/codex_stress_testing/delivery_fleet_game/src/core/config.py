"""Configuration bootstrap utilities."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Resolved filesystem paths and base configuration."""

    project_root: Path
    data_dir: Path
    savegame_dir: Path
    snapshot_dir: Path
    starting_balance: float = 100_000.0
    package_manifest_pattern: str = "packages_day{day}.json"
    vehicles_file: Path = field(init=False)
    map_file: Path = field(init=False)
    template_save_file: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "vehicles_file", self.data_dir / "vehicles.json")
        object.__setattr__(self, "map_file", self.data_dir / "map.json")
        object.__setattr__(self, "template_save_file", self.data_dir / "savegame_template.json")


def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    savegame_dir = project_root / "savegames"
    snapshot_dir = project_root / "snapshots"
    savegame_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        savegame_dir=savegame_dir,
        snapshot_dir=snapshot_dir,
    )
