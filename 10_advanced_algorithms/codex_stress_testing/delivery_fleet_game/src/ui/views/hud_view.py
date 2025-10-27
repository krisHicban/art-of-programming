"""Heads-up display for KPIs and agent comparisons."""

from __future__ import annotations

from typing import Dict, List

import pygame

from ui import theme
from ui.widgets.panel import draw_panel


class HUDView:
    """Renders panels with balance, daily stats, and agent metrics."""

    def __init__(self, surface: pygame.Surface, bounds: pygame.Rect) -> None:
        self.surface = surface
        self.bounds = bounds
        self.font_title = pygame.font.SysFont(theme.FONT_DEFAULT, theme.FONT_SIZE_TITLE)
        self.font_body = pygame.font.SysFont(theme.FONT_DEFAULT, theme.FONT_SIZE_BODY)

    def render(self, balance: float, latest_summary: Dict, agent_history: List[Dict]) -> None:
        draw_panel(self.surface, self.bounds, border_radius=12)
        self._render_balance(balance)
        self._render_summary(latest_summary)
        self._render_agent_history(agent_history)

    def _render_balance(self, balance: float) -> None:
        title = self.font_title.render("Balance", True, theme.TEXT_PRIMARY)
        value = self.font_title.render(f"${balance:,.2f}", True, theme.ACCENT_GREEN)
        self.surface.blit(title, (self.bounds.left + 20, self.bounds.top + 20))
        self.surface.blit(value, (self.bounds.left + 20, self.bounds.top + 50))

    def _render_summary(self, summary: Dict) -> None:
        if not summary:
            return
        y = self.bounds.top + 110
        entries = (
            ("Revenue", f"${summary.get('revenue', 0):,.2f}"),
            ("Costs", f"${summary.get('costs', 0):,.2f}"),
            ("Profit", f"${summary.get('profit', 0):,.2f}"),
            ("Delivered", str(summary.get('packages_delivered', 0))),
        )
        for label, value in entries:
            text = self.font_body.render(f"{label}: {value}", True, theme.TEXT_MUTED)
            self.surface.blit(text, (self.bounds.left + 20, y))
            y += 24

    def _render_agent_history(self, agent_history: List[Dict]) -> None:
        if not agent_history:
            return
        title = self.font_body.render("Recent Agent Runs", True, theme.TEXT_PRIMARY)
        self.surface.blit(title, (self.bounds.left + 20, self.bounds.top + 220))

        column_header = self.font_body.render("Agent    Profit    Distance    Unassigned", True, theme.TEXT_MUTED)
        self.surface.blit(column_header, (self.bounds.left + 20, self.bounds.top + 244))

        y = self.bounds.top + 268
        for run in agent_history[-3:]:
            profit = f"${run['total_profit']:.0f}"
            distance = f"{run['total_distance']:.1f} km"
            unassigned = run['packages_unassigned']
            status_color = theme.ACCENT_GREEN if run.get("success") else theme.ACCENT_ORANGE
            agent_text = self.font_body.render(f"{run['agent_name']:<8} {profit:<10} {distance:<12} {unassigned}", True, theme.TEXT_MUTED)
            indicator_x = self.bounds.left + 14
            pygame.draw.circle(self.surface, status_color, (indicator_x, y + 6), 4)
            self.surface.blit(agent_text, (self.bounds.left + 24, y))
            y += 24
