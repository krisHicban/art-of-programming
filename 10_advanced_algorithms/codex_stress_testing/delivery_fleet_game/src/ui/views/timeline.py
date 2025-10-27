"""Event timeline rendering."""

from __future__ import annotations

from typing import List, Optional

import pygame

from ui import theme
from ui.widgets.panel import draw_panel


class TimelineView:
    """Displays chronological events for the selected day."""

    def __init__(self, surface: pygame.Surface, bounds: pygame.Rect) -> None:
        self.surface = surface
        self.bounds = bounds
        self.font = pygame.font.SysFont(theme.FONT_DEFAULT, theme.FONT_SIZE_BODY)

    def render(self, events: List[dict], active_index: Optional[int] = None) -> None:
        draw_panel(self.surface, self.bounds, border_radius=12)
        title = self.font.render("Event Timeline", True, theme.TEXT_PRIMARY)
        self.surface.blit(title, (self.bounds.left + 20, self.bounds.top + 12))

        y = self.bounds.top + 40
        max_items = 6
        trimmed = events[-max_items:]
        start_index = len(events) - len(trimmed)
        for idx, event in enumerate(trimmed):
            label = f"{event.get('phase', '').title()}: {event.get('event_type', '')}"
            is_active = active_index is not None and (start_index + idx) == active_index
            color = theme.ACCENT_ORANGE if is_active else theme.ACCENT_BLUE
            text_color = theme.TEXT_PRIMARY if is_active else theme.TEXT_MUTED
            pygame.draw.circle(self.surface, color, (self.bounds.left + 14, y + 6), 4)
            text = self.font.render(label, True, text_color)
            self.surface.blit(text, (self.bounds.left + 28, y))
            y += 24
