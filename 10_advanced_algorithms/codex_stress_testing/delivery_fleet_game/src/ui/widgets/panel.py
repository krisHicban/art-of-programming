"""Simple panel helper for drawing rounded backgrounds."""

import pygame

from ui import theme


def draw_panel(surface: pygame.Surface, rect: pygame.Rect, border_radius: int = 8) -> None:
    pygame.draw.rect(surface, theme.PANEL_BG, rect, border_radius=border_radius)
    pygame.draw.rect(surface, theme.PANEL_BORDER, rect, width=1, border_radius=border_radius)
