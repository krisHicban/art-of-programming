"""Console-based interface for Phase 1 manual planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from core.assignment import Assignment


@dataclass
class MenuOption:
    key: str
    description: str


class ConsoleInterface:
    """Simple text UI guiding the admin through the day cycle."""

    def __init__(self) -> None:
        self._options: List[MenuOption] = [
            MenuOption("1", "View pending packages"),
            MenuOption("2", "Assign package to vehicle"),
            MenuOption("3", "Review assignments"),
            MenuOption("4", "Unassign package"),
            MenuOption("5", "Run planning agent (auto assign)"),
            MenuOption("6", "Validate & simulate day"),
            MenuOption("Q", "End planning without running day"),
        ]

    def display_header(self, day: int, balance: float) -> None:
        print("=" * 60)
        print(f"DAY {day} :: Balance ${balance:,.2f}")
        print("=" * 60)

    def display_menu(self) -> None:
        for option in self._options:
            print(f"[{option.key}] {option.description}")

    def display_assignments(self, assignments: Sequence[Assignment]) -> None:
        print("Current assignments:")
        empty = True
        for assignment in assignments:
            if not assignment.packages:
                continue
            empty = False
            print(
                f"- Vehicle {assignment.vehicle.id} ({assignment.vehicle.type}) "
                f"=> {len(assignment.packages)} packages | "
                f"{assignment.total_volume:.2f} m3 | ${assignment.total_revenue:.2f}"
            )
            for pkg in assignment.packages:
                dest = f"({pkg.destination[0]:.1f}, {pkg.destination[1]:.1f})"
                print(
                    f"    · {pkg.id} | {pkg.volume_m3:.2f} m3 | ${pkg.payment_received:.2f} | {dest}"
                )
        if empty:
            print("No packages assigned yet.")

    @staticmethod
    def display_packages(packages) -> None:
        if not packages:
            print("No pending packages.")
            return
        print("Pending packages:")
        for pkg in packages:
            dest = f"({pkg.destination[0]:.1f}, {pkg.destination[1]:.1f})"
            print(
                f"- {pkg.id} | {pkg.volume_m3:.2f} m3 | ${pkg.payment_received:.2f} | {dest}"
            )

    @staticmethod
    def prompt(message: str) -> str:
        return input(f"{message}: ").strip()

    @staticmethod
    def show_message(message: str) -> None:
        print(f"[INFO] {message}")

    @staticmethod
    def show_error(message: str) -> None:
        print(f"[ERROR] {message}")

    @staticmethod
    def confirm(message: str) -> bool:
        answer = input(f"{message} ").strip().lower()
        return answer.startswith("y")

    @staticmethod
    def display_agents(agents) -> None:
        print("Available agents:")
        for index, agent in enumerate(agents, start=1):
            print(f"[{index}] {agent.name}")
