"""Greedy agent implementing a simple first-fit decreasing strategy."""

from __future__ import annotations

from typing import Dict, List

from .base_agent import AgentContext, AgentPlan, BaseAgent
from models.package import Package
from models.vehicle import Vehicle


class GreedyAgent(BaseAgent):
    """Assign packages by filling the largest vehicles first."""

    name = "Greedy Agent"

    def plan(self, context: AgentContext) -> AgentPlan:
        vehicles_sorted = sorted(context.vehicles, key=lambda v: v.capacity_m3, reverse=True)
        remaining_capacity: Dict[str, float] = {vehicle.id: vehicle.capacity_m3 for vehicle in vehicles_sorted}
        assignments: Dict[str, List[Package]] = {vehicle.id: [] for vehicle in vehicles_sorted}
        unassigned: List[Package] = []

        for package in sorted(context.pending_packages, key=lambda p: p.volume_m3, reverse=True):
            placed = False
            for vehicle in vehicles_sorted:
                if remaining_capacity[vehicle.id] >= package.volume_m3:
                    assignments[vehicle.id].append(package)
                    remaining_capacity[vehicle.id] -= package.volume_m3
                    placed = True
                    break
            if not placed:
                unassigned.append(package)

        filtered_assignments = {vid: pkgs for vid, pkgs in assignments.items() if pkgs}
        success = len(unassigned) == 0
        notes = "All packages assigned." if success else f"{len(unassigned)} package(s) left unassigned."

        return AgentPlan(
            assignments=filtered_assignments,
            unassigned=unassigned,
            notes=notes,
            success=success,
        )
