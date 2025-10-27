"""JSON loading utilities for seed data and save games."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
import json

from models.map import Location, MapConfig, Point
from models.package import Package
from models.vehicle import Vehicle
from models.game_state import AgentRun, DailySummary, GameEvent, GameState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_data_path(filename: str) -> Path:
    return DATA_DIR / filename


def load_vehicle_catalog(path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    payload = load_json(path or resolve_data_path("vehicles.json"))
    return payload.get("vehicle_types", {})


def build_vehicle(
    vehicle_id: str,
    vehicle_type: str,
    catalog: Mapping[str, Mapping[str, Any]],
) -> Vehicle:
    spec = catalog.get(vehicle_type)
    if spec is None:
        raise KeyError(f"Unknown vehicle type '{vehicle_type}'")
    return Vehicle(
        id=vehicle_id,
        type=vehicle_type,
        capacity_m3=float(spec["capacity_m3"]),
        cost_per_km=float(spec["cost_per_km"]),
        purchase_price=float(spec["purchase_price"]),
        max_range_km=float(spec["max_range_km"]) if spec.get("max_range_km") is not None else None,
    )


def load_map_config(path: Path | None = None) -> MapConfig:
    payload = load_json(path or resolve_data_path("map.json"))
    locations = [
        Location(
            id=loc["id"],
            name=loc["name"],
            x=float(loc["x"]),
            y=float(loc["y"]),
        )
        for loc in payload.get("locations", [])
    ]
    depot_dict = payload.get("depot", {"x": 0, "y": 0})
    depot: Point = (float(depot_dict.get("x", 0.0)), float(depot_dict.get("y", 0.0)))
    return MapConfig(
        width=float(payload.get("width", 100.0)),
        height=float(payload.get("height", 100.0)),
        depot=depot,
        locations=locations,
    )


def load_packages_for_day(day: int, path: Path | None = None) -> List[Package]:
    filename = f"packages_day{day}.json"
    payload = load_json(path or resolve_data_path(filename))
    return _packages_from_payload(payload)


def load_packages_from_file(path: Path) -> List[Package]:
    payload = load_json(path)
    return _packages_from_payload(payload)


def _packages_from_payload(payload: Mapping[str, Any]) -> List[Package]:
    packages = []
    for raw in payload.get("packages", []):
        packages.append(package_from_dict(raw, default_day=payload.get("day")))
    return packages


def load_game_state(
    catalog: Mapping[str, Mapping[str, Any]],
    path: Path | None = None,
    starting_balance: float | None = None,
) -> GameState:
    if path is None:
        path = resolve_data_path("savegame_template.json")

    payload = load_json(path)
    state = GameState(
        current_day=int(payload.get("current_day", 1)),
        balance=float(payload.get("balance", starting_balance or 100_000.0)),
    )
    for entry in payload.get("fleet", []):
        vehicle = build_vehicle(
            vehicle_id=entry["id"],
            vehicle_type=entry["type"],
            catalog=catalog,
        )
        state.fleet.append(vehicle)

    summaries = [
        DailySummary(
            day=int(summary["day"]),
            packages_delivered=int(summary["packages_delivered"]),
            revenue=float(summary["revenue"]),
            costs=float(summary["costs"]),
            profit=float(summary["profit"]),
        )
        for summary in payload.get("history", [])
    ]
    state.daily_history.extend(summaries)

    state.packages_pending.extend(packages_from_iterable(payload.get("packages_pending", [])))
    state.packages_in_transit.extend(packages_from_iterable(payload.get("packages_in_transit", [])))
    state.packages_delivered.extend(packages_from_iterable(payload.get("packages_delivered", [])))

    for run in payload.get("agent_history", []):
        state.agent_history.append(
            AgentRun(
                day=int(run["day"]),
                agent_name=run["agent_name"],
                success=bool(run.get("success", True)),
                packages_assigned=int(run.get("packages_assigned", 0)),
                packages_unassigned=int(run.get("packages_unassigned", 0)),
                total_distance=float(run.get("total_distance", 0.0)),
                total_revenue=float(run.get("total_revenue", 0.0)),
                total_cost=float(run.get("total_cost", 0.0)),
                total_profit=float(run.get("total_profit", 0.0)),
                notes=run.get("notes"),
            )
        )

    for event in payload.get("events", []):
        state.events.append(
            GameEvent(
                timestamp=event.get("timestamp", ""),
                day=int(event.get("day", state.current_day)),
                phase=event.get("phase", "unknown"),
                event_type=event.get("event_type", "unknown"),
                description=event.get("description", ""),
                payload=event.get("payload", {}),
            )
        )
    return state


def save_game_state(state: GameState, path: Path) -> None:
    serializable = {
        "current_day": state.current_day,
        "balance": state.balance,
        "fleet": [asdict(vehicle) for vehicle in state.fleet],
        "history": [asdict(summary) for summary in state.daily_history],
        "packages_pending": [package_to_dict(pkg) for pkg in state.packages_pending],
        "packages_in_transit": [package_to_dict(pkg) for pkg in state.packages_in_transit],
        "packages_delivered": [package_to_dict(pkg) for pkg in state.packages_delivered],
        "agent_history": [asdict(run) for run in state.agent_history],
        "events": [asdict(event) for event in state.events],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def package_from_dict(raw: Mapping[str, Any], default_day: Optional[int] = None) -> Package:
    destination = raw.get("destination", {})
    dest_point: Point = (float(destination.get("x", 0.0)), float(destination.get("y", 0.0)))
    origin = raw.get("origin", {"x": 0.0, "y": 0.0})
    origin_point: Point = (float(origin.get("x", 0.0)), float(origin.get("y", 0.0)))
    return Package(
        id=raw["id"],
        volume_m3=float(raw["volume_m3"]),
        payment_received=float(raw.get("payment_received", raw.get("payment", 0.0))),
        destination=dest_point,
        origin=origin_point,
        priority=raw.get("priority"),
        weight_kg=raw.get("weight_kg"),
        received_date=raw.get("received_date", default_day),
        delivery_deadline=raw.get("delivery_deadline"),
    )


def package_to_dict(package: Package) -> Dict[str, Any]:
    return {
        "id": package.id,
        "volume_m3": package.volume_m3,
        "payment_received": package.payment_received,
        "destination": {"x": package.destination[0], "y": package.destination[1]},
        "origin": {"x": package.origin[0], "y": package.origin[1]},
        "priority": package.priority,
        "weight_kg": package.weight_kg,
        "received_date": package.received_date,
        "delivery_deadline": package.delivery_deadline,
    }


def packages_from_iterable(payload: Iterable[Mapping[str, Any]]) -> List[Package]:
    return [package_from_dict(raw) for raw in payload]
