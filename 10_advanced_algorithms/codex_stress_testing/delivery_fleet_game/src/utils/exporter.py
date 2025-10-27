"""Utilities for exporting snapshots for visualization layers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional

from models.game_state import GameState
from models.map import MapConfig
from models.route import Route
from utils.data_loader import package_to_dict


def _route_to_dict(route: Route) -> Dict[str, object]:
    return {
        "vehicle_id": route.vehicle_id,
        "package_ids": list(route.package_ids),
        "stops": [[point[0], point[1]] for point in route.stops],
        "total_distance_km": route.total_distance_km,
        "total_cost": route.total_cost,
        "total_revenue": route.total_revenue,
        "total_volume": route.total_volume,
        "is_valid": route.is_valid,
    }


def build_state_snapshot(
    state: GameState,
    map_config: MapConfig,
    routes: Optional[Iterable[Route]] = None,
) -> Dict[str, object]:
    """Prepare a serializable snapshot of the current game state."""
    snapshot: Dict[str, object] = {
        "current_day": state.current_day,
        "balance": state.balance,
        "fleet": [asdict(vehicle) for vehicle in state.fleet],
        "packages_pending": [package_to_dict(pkg) for pkg in state.packages_pending],
        "packages_delivered": [package_to_dict(pkg) for pkg in state.packages_delivered],
        "daily_history": [asdict(summary) for summary in state.daily_history],
        "agent_history": [asdict(run) for run in state.agent_history],
        "events": [asdict(event) for event in state.events],
        "map": {
            "width": map_config.width,
            "height": map_config.height,
            "depot": {"x": map_config.depot[0], "y": map_config.depot[1]},
            "locations": [
                {"id": loc.id, "name": loc.name, "x": loc.x, "y": loc.y}
                for loc in map_config.locations
            ],
        },
    }
    if routes:
        snapshot["routes"] = [_route_to_dict(route) for route in routes]
    else:
        snapshot["routes"] = []
    return snapshot


def export_state_snapshot(
    state: GameState,
    map_config: MapConfig,
    path: Path,
    routes: Optional[Iterable[Route]] = None,
) -> None:
    """Write the current game snapshot to disk."""
    payload = build_state_snapshot(state, map_config=map_config, routes=routes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
