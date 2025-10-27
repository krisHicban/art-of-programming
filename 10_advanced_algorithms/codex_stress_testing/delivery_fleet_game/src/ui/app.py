"""Pygame application bootstrap for visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame

from core.config import load_settings
from ui import theme
from ui.controllers import SnapshotData, load_snapshot
from ui.views.map_view import MapView
from ui.views.hud_view import HUDView
from ui.views.timeline import TimelineView


class VisualizationApp:
    """High-level controller managing the visualization loop."""

    def __init__(self, snapshot_path: Optional[Path] = None) -> None:
        self.settings = load_settings()
        self.snapshot_path = snapshot_path
        self.snapshot: Optional[SnapshotData] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._running = False
        self._map_view: Optional[MapView] = None
        self._hud_view: Optional[HUDView] = None
        self._timeline_view: Optional[TimelineView] = None
        self._font_header: Optional[pygame.font.Font] = None
        self._font_small: Optional[pygame.font.Font] = None
        self._snapshots: List[Path] = []
        self._snapshot_index: int = 0
        self._route_segments: Dict[str, List[Dict[str, float]]] = {}
        self._route_totals: Dict[str, float] = {}
        self._playback_time: float = 0.0
        self._playback_total: float = 1.0
        self._playback_speed: float = 20.0
        self._playback_active: bool = True

    def _resolve_latest_snapshot(self) -> Optional[Path]:
        snapshots = self._load_snapshot_paths()
        return snapshots[-1] if snapshots else None

    def _load_snapshot_paths(self) -> List[Path]:
        self._snapshots = sorted(self.settings.snapshot_dir.glob("*.json"))
        if self.snapshot_path and self.snapshot_path.exists():
            try:
                self._snapshot_index = self._snapshots.index(self.snapshot_path)
            except ValueError:
                self._snapshots.append(self.snapshot_path)
                self._snapshots.sort()
                self._snapshot_index = self._snapshots.index(self.snapshot_path)
        elif self._snapshots:
            self._snapshot_index = len(self._snapshots) - 1
            self.snapshot_path = self._snapshots[self._snapshot_index]
        return self._snapshots

    def load_snapshot(self) -> None:
        if not self.snapshot_path:
            self.snapshot_path = self._resolve_latest_snapshot()
        if not self.snapshot_path or not self.snapshot_path.exists():
            raise FileNotFoundError("No snapshot available. Run a simulation day first.")
        self.snapshot = load_snapshot(self.snapshot_path)
        self._prepare_route_playback()

    def run(self) -> None:
        """Start the Pygame loop (placeholder implementation)."""
        self.load_snapshot()
        pygame.init()
        screen = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption("Delivery Fleet Visualization")
        self._load_snapshot_paths()
        self._clock = pygame.time.Clock()
        self._running = True
        self._font_header = pygame.font.SysFont(theme.FONT_DEFAULT, 28)
        self._create_views(screen)

        while self._running:
            dt = self._clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self._running = False
                if event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)

            self._update_playback(dt)
            self._render(screen)
            pygame.display.flip()

        pygame.quit()

    def _create_views(self, screen: pygame.Surface) -> None:
        map_rect = pygame.Rect(20, 20, 820, 520)
        hud_rect = pygame.Rect(860, 20, 400, 320)
        timeline_rect = pygame.Rect(20, 560, 1240, 120)

        self._map_view = MapView(screen, map_rect)
        self._hud_view = HUDView(screen, hud_rect)
        self._timeline_view = TimelineView(screen, timeline_rect)

        if self.snapshot:
            map_info = self.snapshot.raw.get("map", {"width": 100, "height": 100})
            width = float(map_info.get("width", 100))
            height = float(map_info.get("height", 100))
            self._map_view.set_map_dimensions(width, height)

    def _render(self, screen: pygame.Surface) -> None:
        screen.fill(theme.PRIMARY_BG)
        if not self.snapshot or not self._map_view or not self._hud_view or not self._timeline_view:
            return

        map_info = self.snapshot.raw.get("map", {"depot": {"x": 0, "y": 0}, "width": 100, "height": 100})
        depot_point = (
            float(map_info.get("depot", {}).get("x", 0.0)),
            float(map_info.get("depot", {}).get("y", 0.0)),
        )
        pending_packages = self.snapshot.raw.get("packages_pending", [])
        delivered_packages = self.snapshot.raw.get("packages_delivered", [])
        routes_dict = self._collect_routes()
        vehicle_colors = self._vehicle_color_map()
        self._map_view.render(depot_point, pending_packages, delivered_packages, routes_dict, vehicle_colors)
        self._render_vehicle_markers(vehicle_colors)
        self._render_legend(screen, (40, 520))

        latest_summary = self.snapshot.raw.get("daily_history", [])[-1] if self.snapshot.raw.get("daily_history") else {}
        agent_history = self.snapshot.agent_history
        balance = float(self.snapshot.raw.get("balance", 0.0))
        self._hud_view.render(balance, latest_summary, agent_history)

        day = self.snapshot.raw.get("current_day", 0)
        day_events = [event for event in self.snapshot.events if event.get("day") == day] or self.snapshot.events
        active_event_index = self._active_event_index(day_events)
        self._timeline_view.render(day_events, active_index=active_event_index)

        header_text = self._font_header.render(
            f"Day {day} Snapshot - Pending {len(pending_packages)} | Delivered {len(delivered_packages)}",
            True,
            theme.TEXT_PRIMARY,
        )
        screen.blit(header_text, (20, 0))
        if self._font_header:
            status = "►" if self._playback_active else "❚❚"
            footer = self._font_header.render(
                f"{status} [Space] Play/Pause   [Left/Right] Change day   [R] Reload   [Esc] Quit   Snapshot {self._snapshot_index + 1}/{max(1, len(self._snapshots))}",
                True,
                theme.TEXT_MUTED,
            )
            screen.blit(footer, (20, 700))

    def _collect_routes(self) -> Dict[str, List[Tuple[float, float]]]:
        routes: Dict[str, List[Tuple[float, float]]] = {}
        for route in self.snapshot.raw.get("routes", []):
            vehicle_id = route.get("vehicle_id")
            stops = route.get("stops", [])
            if not vehicle_id or not stops:
                continue
            routes[vehicle_id] = [(float(point[0]), float(point[1])) for point in stops]
        return routes

    def _vehicle_color_map(self) -> Dict[str, Tuple[int, int, int]]:
        palette: List[Tuple[int, int, int]] = [
            theme.ACCENT_BLUE,
            theme.ACCENT_ORANGE,
            theme.ACCENT_GREEN,
            (155, 89, 182),
            (231, 76, 60),
        ]
        mapping: Dict[str, Tuple[int, int, int]] = {}
        fleet = self.snapshot.raw.get("fleet", [])
        for idx, vehicle in enumerate(fleet):
            vehicle_id = vehicle.get("id")
            if not vehicle_id:
                continue
            mapping[vehicle_id] = palette[idx % len(palette)]
        return mapping

    def _render_legend(self, screen: pygame.Surface, origin: Tuple[int, int]) -> None:
        if not self._font_header:
            return
        font_small = pygame.font.SysFont(theme.FONT_DEFAULT, 14)
        x, y = origin
        legend_items = [
            (theme.ACCENT_BLUE, "Depot"),
            (theme.ACCENT_ORANGE, "Pending package"),
            (theme.ACCENT_GREEN, "Delivered package"),
        ]
        for color, label in legend_items:
            pygame.draw.circle(screen, color, (x, y), 6)
            text = font_small.render(label, True, theme.TEXT_MUTED)
            screen.blit(text, (x + 12, y - 7))
            x += 140

    def _handle_key(self, key: int) -> None:
        if key in (pygame.K_LEFT, pygame.K_a):
            self._move_snapshot(-1)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self._move_snapshot(1)
        elif key in (pygame.K_r,):
            self._reload_snapshots()
        elif key in (pygame.K_SPACE,):
            self._playback_active = not self._playback_active

    def _move_snapshot(self, delta: int) -> None:
        if not self._snapshots:
            return
        self._snapshot_index = max(0, min(self._snapshot_index + delta, len(self._snapshots) - 1))
        self.snapshot_path = self._snapshots[self._snapshot_index]
        self.load_snapshot()
        if self._map_view and self.snapshot:
            map_info = self.snapshot.raw.get("map", {"width": 100, "height": 100})
            width = float(map_info.get("width", 100))
            height = float(map_info.get("height", 100))
            self._map_view.set_map_dimensions(width, height)

    def _reload_snapshots(self) -> None:
        before = self._snapshots
        self._load_snapshot_paths()
        if self._snapshots != before:
            self._snapshot_index = min(self._snapshot_index, len(self._snapshots) - 1)
            self.snapshot_path = self._snapshots[self._snapshot_index]
            self.load_snapshot()

    def _update_playback(self, dt: float) -> None:
        if not self._route_totals or self._playback_total <= 0:
            return
        if self._playback_active:
            self._playback_time = (self._playback_time + dt * self._playback_speed) % self._playback_total

    def _render_vehicle_markers(self, vehicle_colors: Dict[str, Tuple[int, int, int]]) -> None:
        if not self._route_segments or not self._map_view:
            return
        for vehicle_id, segments in self._route_segments.items():
            total = self._route_totals.get(vehicle_id, 0.0)
            if total <= 0 or not segments:
                continue
            progress = self._playback_time % total if total > 0 else 0.0
            point = self._interpolate_segment(segments, progress)
            if point is None:
                continue
            color = vehicle_colors.get(vehicle_id, theme.ACCENT_GREEN)
            self._map_view.draw_vehicle_marker(point, color)

    def _interpolate_segment(self, segments: List[Dict[str, float]], progress: float) -> Optional[Tuple[float, float]]:
        if not segments:
            return None
        for segment in segments:
            if progress <= segment["cumulative_end"]:
                length = segment["length"] or 1e-6
                t = (progress - segment["cumulative_start"]) / length
                t = max(0.0, min(1.0, t))
                x = segment["start_x"] + (segment["end_x"] - segment["start_x"]) * t
                y = segment["start_y"] + (segment["end_y"] - segment["start_y"]) * t
                return (x, y)
        last = segments[-1]
        return (last["end_x"], last["end_y"])

    def _active_event_index(self, events: List[dict]) -> Optional[int]:
        if not events:
            return None
        if self._playback_total <= 0:
            return len(events) - 1
        ratio = min(1.0, self._playback_time / self._playback_total)
        index = int(ratio * len(events))
        return min(len(events) - 1, index)

    def _prepare_route_playback(self) -> None:
        self._route_segments = {}
        self._route_totals = {}
        if not self.snapshot:
            self._playback_total = 1.0
            return

        for route in self.snapshot.raw.get("routes", []):
            vehicle_id = route.get("vehicle_id")
            stops = route.get("stops", [])
            if not vehicle_id or len(stops) < 2:
                continue
            segments: List[Dict[str, float]] = []
            total = 0.0
            prev_point: Optional[Tuple[float, float]] = None
            for point in stops:
                current = (float(point[0]), float(point[1]))
                if prev_point is not None:
                    length = ((current[0] - prev_point[0]) ** 2 + (current[1] - prev_point[1]) ** 2) ** 0.5
                    segment = {
                        "start_x": prev_point[0],
                        "start_y": prev_point[1],
                        "end_x": current[0],
                        "end_y": current[1],
                        "length": length,
                        "cumulative_start": total,
                        "cumulative_end": total + length,
                    }
                    segments.append(segment)
                    total += length
                prev_point = current
            self._route_segments[vehicle_id] = segments
            self._route_totals[vehicle_id] = total

        self._playback_total = max(self._route_totals.values(), default=0.0)
        self._playback_time = 0.0
        self._playback_active = True


def preview_latest_snapshot() -> None:
    """Convenience function to launch the visualization with the newest snapshot."""
    app = VisualizationApp()
    app.run()
