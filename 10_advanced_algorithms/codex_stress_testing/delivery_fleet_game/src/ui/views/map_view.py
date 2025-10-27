"""Map rendering surface."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pygame

from ui import theme
from ui.widgets.panel import draw_panel

Point = Tuple[float, float]


class MapView:
    """Draws the depot, package destinations, and vehicle routes."""

    def __init__(self, surface: pygame.Surface, bounds: pygame.Rect) -> None:
        self.surface = surface
        self.bounds = bounds
        self._grid_color = (60, 65, 75)
        self._map_width = 100.0
        self._map_height = 100.0
        self._center = (bounds.centerx, bounds.centery)

    def set_map_dimensions(self, width: float, height: float) -> None:
        self._map_width = max(width, 1.0)
        self._map_height = max(height, 1.0)

    def render(
        self,
        depot: Point,
        pending_packages: List[Dict],
        delivered_packages: List[Dict],
        routes: Dict[str, List[Point]],
        vehicle_colors: Dict[str, Tuple[int, int, int]],
    ) -> None:
        """Render the map contents."""
        draw_panel(self.surface, self.bounds, border_radius=12)
        self._draw_grid()
        self._draw_depot(depot)
        self._draw_packages(pending_packages, theme.ACCENT_ORANGE)
        self._draw_packages(delivered_packages, theme.ACCENT_GREEN)
        self._draw_routes(routes, vehicle_colors)

    def _draw_grid(self) -> None:
        spacing = 40
        for x in range(self.bounds.left, self.bounds.right, spacing):
            pygame.draw.line(self.surface, self._grid_color, (x, self.bounds.top), (x, self.bounds.bottom))
        for y in range(self.bounds.top, self.bounds.bottom, spacing):
            pygame.draw.line(self.surface, self._grid_color, (self.bounds.left, y), (self.bounds.right, y))

    def _draw_depot(self, depot: Point) -> None:
        pygame.draw.circle(self.surface, theme.ACCENT_BLUE, self._project(depot), 10)

    def _draw_packages(self, packages: List[Dict], color: Tuple[int, int, int]) -> None:
        for pkg in packages:
            if "destination" not in pkg:
                continue
            destination = pkg["destination"]
            point = (float(destination.get("x", 0.0)), float(destination.get("y", 0.0)))
            pygame.draw.circle(self.surface, color, self._project(point), 5)

    def _draw_routes(self, routes: Dict[str, List[Point]], vehicle_colors: Dict[str, Tuple[int, int, int]]) -> None:
        for vehicle_id, points in routes.items():
            color = vehicle_colors.get(vehicle_id, theme.ACCENT_GREEN)
            if len(points) < 2:
                continue
            scaled_points = [self._project(point) for point in points]
            pygame.draw.lines(self.surface, color, False, scaled_points, width=2)

    def project(self, point: Point) -> Tuple[int, int]:
        return self._project(point)

    def draw_vehicle_marker(self, point: Point, color: Tuple[int, int, int]) -> None:
        pygame.draw.circle(self.surface, color, self._project(point), 7)

    def _project(self, point: Point) -> Tuple[int, int]:
        scale_x = self.bounds.width / (self._map_width * 2)
        scale_y = self.bounds.height / (self._map_height * 2)
        screen_x = int(self._center[0] + point[0] * scale_x)
        screen_y = int(self._center[1] - point[1] * scale_y)
        return screen_x, screen_y
