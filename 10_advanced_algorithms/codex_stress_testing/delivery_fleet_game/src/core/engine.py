"""Top-level game engine orchestrating the day cycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from agents.base_agent import AgentContext, AgentPlan, BaseAgent
from agents.greedy_agent import GreedyAgent
from core.assignment import Assignment, AssignmentManager
from core.cli import ConsoleInterface
from core.config import Settings, load_settings
from core.ledger import Ledger
from core.route_builder import build_route
from core.validator import ValidationError, validate_routes
from models.game_state import AgentRun, DailySummary, GameState
from models.map import MapConfig
from models.package import Package
from models.route import Route
from models.vehicle import Vehicle
from utils import data_loader
from utils.exporter import export_state_snapshot
from utils.metrics import RouteMetrics, aggregate_routes

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover - tabulate should be available via requirements
    tabulate = None  # type: ignore


@dataclass
class LoadedAssets:
    """Container for preloaded resources."""

    catalog: Dict[str, Dict[str, Any]]
    map_config: MapConfig
    game_state: GameState


class GameEngine:
    """Coordinates loading data, running the planning cycle, and persisting results."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or load_settings()
        self.assets: Optional[LoadedAssets] = None
        self.console = ConsoleInterface()
        self.agents: List[BaseAgent] = [GreedyAgent()]
        self._loaded_save_path: Optional[Path] = None

    def initialize(self) -> None:
        """Load configuration, map, vehicle catalog, and base game state."""
        catalog = data_loader.load_vehicle_catalog(self.settings.vehicles_file)
        map_config = data_loader.load_map_config(self.settings.map_file)
        latest_save = self._find_latest_save()
        if latest_save is not None:
            state = data_loader.load_game_state(
                catalog=catalog,
                path=latest_save,
                starting_balance=self.settings.starting_balance,
            )
            self._loaded_save_path = latest_save
        else:
            state = data_loader.load_game_state(
                catalog=catalog,
                path=self.settings.template_save_file,
                starting_balance=self.settings.starting_balance,
            )
            self._loaded_save_path = None
        self.assets = LoadedAssets(catalog=catalog, map_config=map_config, game_state=state)

    def run(self) -> None:
        """Execute a single day loop (placeholder for Phase 1 implementation)."""
        if self.assets is None:
            self.initialize()
        assert self.assets is not None
        state = self.assets.game_state
        map_config = self.assets.map_config
        vehicles = {vehicle.id: vehicle for vehicle in state.fleet}
        new_packages = data_loader.load_packages_for_day(state.current_day)
        existing_ids = {pkg.id for pkg in state.packages_pending}
        state.packages_pending.extend(pkg for pkg in new_packages if pkg.id not in existing_ids)
        state.log_event(
            phase="planning",
            event_type="day_start",
            description=f"Day {state.current_day} planning phase opened.",
            payload={
                "new_packages": len(new_packages),
                "pending_total": len(state.packages_pending),
                "balance": state.balance,
            },
        )

        assignment_manager = AssignmentManager(state.fleet)

        if self._loaded_save_path:
            self.console.show_message(
                f"Loaded save from {self._loaded_save_path.name}. Resuming on day {state.current_day}."
            )

        while True:
            self.console.display_header(state.current_day, state.balance)
            self.console.display_menu()
            choice = self.console.prompt("Select option").upper()

            if choice == "1":
                self.console.display_packages(state.packages_pending)
            elif choice == "2":
                self._assign_package(state, vehicles, assignment_manager)
            elif choice == "3":
                self.console.display_assignments(assignment_manager.summary())
            elif choice == "4":
                self._unassign_package(state, assignment_manager)
            elif choice == "5":
                self._run_agent_menu(state, map_config, assignment_manager, vehicles)
            elif choice == "6":
                if not assignment_manager.has_assignments():
                    self.console.show_error("No assignments available. Assign manually or run an agent first.")
                    continue
                if self._simulate_day(state, map_config, assignment_manager, vehicles):
                    break
            elif choice == "Q":
                self._return_all_assignments_to_pending(state, assignment_manager)
                self._save_progress(state, filename="autosave.json")
                autosave_snapshot = self.settings.snapshot_dir / "autosave.json"
                export_state_snapshot(state, map_config, autosave_snapshot)
                self.console.show_message(f"Snapshot exported to {autosave_snapshot}")
                state.log_event(
                    phase="planning",
                    event_type="autosave_exit",
                    description="Planning ended without executing day.",
                    payload={
                        "pending_packages": len(state.packages_pending),
                        "snapshot": str(autosave_snapshot.name),
                    },
                )
                self.console.show_message("Progress saved. Exiting without running day.")
                break
            else:
                self.console.show_error("Unknown option.")

    def _assign_package(
        self,
        state: GameState,
        vehicles: Dict[str, Vehicle],
        assignment_manager: AssignmentManager,
    ) -> None:
        if not state.packages_pending:
            self.console.show_message("No pending packages to assign.")
            return

        package_id = self.console.prompt("Enter package id")
        package = self._pop_pending_package(state, package_id)
        if package is None:
            self.console.show_error(f"Package '{package_id}' not found in pending list.")
            return

        vehicle_id = self.console.prompt("Assign to vehicle id")
        vehicle = vehicles.get(vehicle_id)
        if vehicle is None:
            self.console.show_error(f"Vehicle '{vehicle_id}' not found.")
            state.packages_pending.append(package)
            return

        assignment_manager.add_package(vehicle_id, package)
        self.console.show_message(
            f"Assigned package {package.id} to vehicle {vehicle_id} "
            f"({package.volume_m3:.2f} m3 | ${package.payment_received:.2f})"
        )
        state.log_event(
            phase="planning",
            event_type="manual_assignment",
            description=f"Package {package.id} assigned to vehicle {vehicle_id}.",
            payload={
                "package_id": package.id,
                "vehicle_id": vehicle_id,
                "volume_m3": package.volume_m3,
                "payment": package.payment_received,
            },
        )

    def _unassign_package(self, state: GameState, assignment_manager: AssignmentManager) -> None:
        package_id = self.console.prompt("Package id to unassign")
        assignment = assignment_manager.find_assignment_for_package(package_id)
        if assignment is None:
            self.console.show_error(f"Package '{package_id}' is not currently assigned.")
            return
        package = assignment_manager.remove_package(assignment.vehicle.id, package_id)
        if package is not None:
            state.packages_pending.append(package)
            self.console.show_message(f"Returned package {package_id} to pending list.")
            state.log_event(
                phase="planning",
                event_type="manual_unassignment",
                description=f"Package {package_id} returned to pending list.",
                payload={"package_id": package_id, "vehicle_id": assignment.vehicle.id},
            )
        else:
            self.console.show_error(f"Package '{package_id}' could not be removed.")

    @staticmethod
    def _pop_pending_package(state: GameState, package_id: str) -> Optional[Package]:
        for index, package in enumerate(state.packages_pending):
            if package.id == package_id:
                return state.packages_pending.pop(index)
        return None

    def _simulate_day(
        self,
        state: GameState,
        map_config: MapConfig,
        assignment_manager: AssignmentManager,
        vehicles: Dict[str, Vehicle],
    ) -> bool:
        if state.packages_pending:
            self.console.show_error("Cannot simulate day while packages remain unassigned.")
            return False

        route_plan = []
        for assignment in assignment_manager.summary():
            if not assignment.packages:
                continue
            route = build_route(
                vehicle=assignment.vehicle,
                packages=assignment.packages,
                depot=map_config.depot,
            )
            route_plan.append((route, assignment))

        if not route_plan:
            self.console.show_error("No routes planned. Assign packages before simulation.")
            return False

        routes = [route for route, _ in route_plan]
        try:
            validate_routes(routes, vehicles, depot=map_config.depot)
        except ValidationError as exc:
            self.console.show_error(str(exc))
            state.log_event(
                phase="planning",
                event_type="validation_error",
                description=str(exc),
                payload={"vehicle_ids": [route.vehicle_id for route in routes]},
            )
            return False

        ledger = Ledger()
        delivered: List[Package] = []

        for route, assignment in route_plan:
            route.is_valid = True
            ledger.record_route(route)
            delivered.extend(list(assignment.packages))
            assignment.vehicle.reset_location()

        completed_day = state.current_day
        state.packages_delivered.extend(delivered)
        revenue = ledger.revenue()
        costs = ledger.expenses()
        profit = ledger.profit()
        summary = DailySummary(
            day=completed_day,
            packages_delivered=len(delivered),
            revenue=revenue,
            costs=costs,
            profit=profit,
        )
        state.record_summary(summary)
        state.packages_in_transit.clear()
        assignment_manager.clear()
        state.packages_pending.clear()
        state.advance_day()

        self._print_route_summary(route_plan)

        self.console.show_message(
            f"Day complete! Revenue ${revenue:,.2f} | Costs ${costs:,.2f} | Profit ${profit:,.2f}"
        )
        self.console.show_message(f"Delivered {len(delivered)} packages.")

        filename = f"day_{completed_day:02d}.json"
        self._save_progress(state, filename=filename, quiet=True)
        self.console.show_message(f"Progress saved to {self.settings.savegame_dir / filename}")
        routes = [route for route, _ in route_plan]
        snapshot_path = self.settings.snapshot_dir / f"day_{completed_day:02d}.json"
        export_state_snapshot(state, map_config, snapshot_path, routes=routes)
        self.console.show_message(f"Snapshot exported to {snapshot_path}")
        state.log_event(
            phase="execution",
            event_type="day_completed",
            description=f"Day {completed_day} completed.",
            payload={
                "revenue": revenue,
                "costs": costs,
                "profit": profit,
                "packages_delivered": len(delivered),
                "save_file": filename,
            },
            day_override=completed_day,
        )
        return True

    def _return_all_assignments_to_pending(
        self,
        state: GameState,
        assignment_manager: AssignmentManager,
    ) -> None:
        existing_ids: Set[str] = {pkg.id for pkg in state.packages_pending}
        for assignment in assignment_manager.summary():
            for package in assignment.packages:
                if package.id not in existing_ids:
                    state.packages_pending.append(package)
                    existing_ids.add(package.id)
            assignment.packages.clear()

    def _save_progress(self, state: GameState, filename: Optional[str] = None, quiet: bool = False) -> None:
        target = self.settings.savegame_dir / (filename or "autosave.json")
        data_loader.save_game_state(state, target)
        if not quiet:
            self.console.show_message(f"Progress saved to {target}")

    def _print_route_summary(self, route_plan: List[Tuple[Route, Assignment]]) -> None:
        rows = []
        for route, assignment in route_plan:
            rows.append(
                [
                    assignment.vehicle.id,
                    assignment.vehicle.type,
                    len(assignment.packages),
                    f"{route.total_volume:.2f}",
                    f"{route.total_distance_km:.2f}",
                    f"${route.total_revenue:.2f}",
                    f"${route.total_cost:.2f}",
                    f"${(route.total_revenue - route.total_cost):.2f}",
                ]
            )
        headers = ["Vehicle", "Type", "#Pkgs", "Volume(m3)", "Distance(km)", "Revenue", "Cost", "Profit"]
        metrics = aggregate_routes([route for route, _ in route_plan])
        if tabulate:
            print(tabulate(rows, headers=headers, tablefmt="github"))
        else:
            print("Route summary:")
            print(" | ".join(headers))
            for row in rows:
                print(" | ".join(str(item) for item in row))
        print(
            f"Totals -> Vehicles: {metrics.vehicle_count}, Packages: {metrics.package_count}, "
            f"Distance: {metrics.total_distance:.2f} km, Revenue: ${metrics.total_revenue:.2f}, "
            f"Cost: ${metrics.total_cost:.2f}, Profit: ${metrics.total_profit:.2f}"
        )

    def _find_latest_save(self) -> Optional[Path]:
        if not self.settings.savegame_dir.exists():
            return None
        candidates = sorted(
            self.settings.savegame_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _run_agent_menu(
        self,
        state: GameState,
        map_config: MapConfig,
        assignment_manager: AssignmentManager,
        vehicles: Dict[str, Vehicle],
    ) -> None:
        if not self.agents:
            self.console.show_error("No agents available.")
            return
        self.console.display_agents(self.agents)
        selection = self.console.prompt("Select agent number")
        try:
            index = int(selection) - 1
        except ValueError:
            self.console.show_error("Invalid selection.")
            return
        if index < 0 or index >= len(self.agents):
            self.console.show_error("Selection out of range.")
            return

        agent = self.agents[index]
        context = AgentContext(
            day=state.current_day,
            balance=state.balance,
            map_config=map_config,
            vehicles=list(vehicles.values()),
            pending_packages=list(state.packages_pending),
        )

        plan = agent.plan(context)
        self._apply_agent_plan(plan, agent, state, assignment_manager, vehicles, map_config)

    def _apply_agent_plan(
        self,
        plan: AgentPlan,
        agent: BaseAgent,
        state: GameState,
        assignment_manager: AssignmentManager,
        vehicles: Dict[str, Vehicle],
        map_config: MapConfig,
    ) -> None:
        self._return_all_assignments_to_pending(state, assignment_manager)
        assignment_manager.clear()

        assigned_ids: Set[str] = set()
        for vehicle_id, packages in plan.assignments.items():
            if vehicle_id not in vehicles:
                self.console.show_error(f"{agent.name} referenced unknown vehicle '{vehicle_id}'.")
                continue
            for package in packages:
                assignment_manager.add_package(vehicle_id, package)
                assigned_ids.add(package.id)

        if assigned_ids:
            state.packages_pending = [pkg for pkg in state.packages_pending if pkg.id not in assigned_ids]

        self.console.show_message(f"{agent.name} completed planning. {plan.notes}")
        if plan.unassigned:
            preview = ", ".join(pkg.id for pkg in plan.unassigned[:5])
            more = "..." if len(plan.unassigned) > 5 else ""
            self.console.show_error(
                f"{len(plan.unassigned)} packages could not be assigned: {preview}{more}"
            )

        summaries = assignment_manager.summary()
        self.console.display_assignments(summaries)

        route_plan: List[Tuple[Route, Assignment]] = []
        for assignment in summaries:
            if not assignment.packages:
                continue
            route = build_route(
                vehicle=assignment.vehicle,
                packages=assignment.packages,
                depot=map_config.depot,
            )
            route_plan.append((route, assignment))

        metrics: RouteMetrics
        if route_plan:
            self._print_route_summary(route_plan)
            metrics = aggregate_routes([route for route, _ in route_plan])
        else:
            metrics = RouteMetrics(
                total_distance=0.0,
                total_revenue=0.0,
                total_cost=0.0,
                total_profit=0.0,
                vehicle_count=0,
                package_count=0,
            )

        state.record_agent_run(
            AgentRun(
                day=state.current_day,
                agent_name=agent.name,
                success=plan.success,
                packages_assigned=metrics.package_count,
                packages_unassigned=len(plan.unassigned),
                total_distance=metrics.total_distance,
                total_revenue=metrics.total_revenue,
                total_cost=metrics.total_cost,
                total_profit=metrics.total_profit,
                notes=plan.notes,
            )
        )
        state.log_event(
            phase="planning",
            event_type="agent_plan",
            description=f"{agent.name} generated a plan.",
            payload={
                "agent": agent.name,
                "success": plan.success,
                "packages_assigned": metrics.package_count,
                "packages_unassigned": len(plan.unassigned),
                "total_distance": metrics.total_distance,
                "total_revenue": metrics.total_revenue,
                "total_cost": metrics.total_cost,
                "total_profit": metrics.total_profit,
            },
        )

        if plan.success:
            if self.console.confirm("Run simulation with this plan now? (y/n)"):
                self._simulate_day(state, map_config, assignment_manager, vehicles)
