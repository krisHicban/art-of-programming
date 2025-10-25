#!/usr/bin/env python3
"""
Delivery Fleet Management System - Pygame GUI Version

Beautiful, interactive visualization of the delivery route optimization game.

Usage:
    python main_pygame.py
"""

import sys
import pygame
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import GameEngine
from src.agents import GreedyAgent, BacktrackingAgent, PruningBacktrackingAgent
from src.ui.constants import *
from src.ui.map_renderer import MapRenderer
from src.ui.components import Button, Panel, StatDisplay, RadioButton, Tooltip
from src.utils import format_game_statistics


class DeliveryFleetApp:
    """
    Main Pygame application for the Delivery Fleet game.

    Handles the game loop, rendering, and user interaction.
    """

    def __init__(self):
        """Initialize the application."""
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("Delivery Fleet Manager - Art of Programming")

        # Create window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True

        # Create surfaces for different areas
        self.map_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))
        self.sidebar_surface = pygame.Surface((SIDEBAR_WIDTH, WINDOW_HEIGHT))

        # Initialize game engine
        data_dir = Path(__file__).parent / "data"
        self.engine = GameEngine(data_dir)

        # Initialize map renderer
        self.map_renderer = MapRenderer(self.map_surface, self.engine.delivery_map)

        # Register agents
        self._register_agents()

        # UI state
        self.selected_agent = "greedy"
        self.tooltip = Tooltip()
        self.current_view = "main"  # main, comparison, stats

        # Create UI components
        self._create_ui_components()

        # Game state
        self.planned_routes = []
        self.package_status = {}  # package_id -> "pending"/"in_transit"/"delivered"

        # Start new game
        self.engine.new_game()

        print("✓ Pygame application initialized!")
        print("✓ Window created: 1200x800")
        print("✓ Ready to play!")

    def _register_agents(self):
        """Register all routing agents."""
        self.engine.register_agent("greedy", GreedyAgent(self.engine.delivery_map))
        self.engine.register_agent("greedy_2opt", GreedyAgent(self.engine.delivery_map, use_2opt=True))
        self.engine.register_agent("backtracking", BacktrackingAgent(self.engine.delivery_map, max_packages=12))
        self.engine.register_agent("pruning_backtracking",
                                   PruningBacktrackingAgent(self.engine.delivery_map, max_packages=15))

    def _create_ui_components(self):
        """Create all UI components."""
        # Panels
        self.stats_panel = Panel(SIDEBAR_X + 10, 110, SIDEBAR_WIDTH - 20, STATS_PANEL_HEIGHT, "GAME STATUS")
        self.agent_panel = Panel(SIDEBAR_X + 10, 320, SIDEBAR_WIDTH - 20, AGENT_PANEL_HEIGHT, "ROUTING AGENTS")
        self.controls_panel = Panel(SIDEBAR_X + 10, 630, SIDEBAR_WIDTH - 20, 160, "CONTROLS")

        # Stat displays
        stat_x = SIDEBAR_X + 30
        self.day_stat = StatDisplay(stat_x, 150, "Day:", "1")
        self.balance_stat = StatDisplay(stat_x + 150, 150, "Balance:", "$100,000")
        self.fleet_stat = StatDisplay(stat_x, 200, "Fleet:", "2 vehicles")
        self.packages_stat = StatDisplay(stat_x + 150, 200, "Pending:", "0 packages")

        # Agent radio buttons
        radio_x = SIDEBAR_X + 40
        radio_y = 360
        self.agent_radios = [
            RadioButton(radio_x, radio_y, "Greedy (Fast)", "agent", "greedy"),
            RadioButton(radio_x, radio_y + 40, "Greedy + 2-opt", "agent", "greedy_2opt"),
            RadioButton(radio_x, radio_y + 80, "Backtracking", "agent", "backtracking"),
            RadioButton(radio_x, radio_y + 120, "Pruning Backtrack", "agent", "pruning_backtracking"),
        ]
        self.agent_radios[0].selected = True  # Default selection

        # Control buttons
        btn_x = SIDEBAR_X + 30
        btn_y = 670
        btn_width = SIDEBAR_WIDTH - 60
        self.buttons = {
            'start_day': Button(btn_x, btn_y, btn_width, BUTTON_HEIGHT, "📦 Start Day", self.on_start_day),
            'plan_routes': Button(btn_x, btn_y + 50, btn_width, BUTTON_HEIGHT, "🧠 Plan Routes", self.on_plan_routes),
            'execute_day': Button(btn_x, btn_y + 100, btn_width // 2 - 5, BUTTON_HEIGHT, "▶️ Execute", self.on_execute_day),
            'next_day': Button(btn_x + btn_width // 2 + 5, btn_y + 100, btn_width // 2 - 5, BUTTON_HEIGHT, "⏭️ Next", self.on_next_day),
        }

        # Set initial button states
        self.buttons['plan_routes'].enabled = False
        self.buttons['execute_day'].enabled = False
        self.buttons['next_day'].enabled = False

    def on_start_day(self):
        """Handle Start Day button click."""
        print("\n[UI] Starting day...")
        self.engine.start_day()
        self.package_status = {pkg.id: "pending" for pkg in self.engine.game_state.packages_pending}
        self.buttons['plan_routes'].enabled = True
        self.update_stats()

    def on_plan_routes(self):
        """Handle Plan Routes button click."""
        print(f"\n[UI] Planning routes with {self.selected_agent}...")
        metrics = self.engine.test_agent(self.selected_agent)

        if metrics:
            self.planned_routes = metrics.get('routes', [])
            self.engine.apply_agent_solution(self.selected_agent)
            self.buttons['execute_day'].enabled = True
            print(f"[UI] Routes planned! Profit: ${metrics['total_profit']:.2f}")

    def on_execute_day(self):
        """Handle Execute Day button click."""
        print("\n[UI] Executing day...")
        self.engine.execute_day(self.selected_agent)

        # Update package statuses
        for pkg in self.engine.game_state.packages_delivered:
            if pkg.id in self.package_status:
                self.package_status[pkg.id] = "delivered"

        self.buttons['next_day'].enabled = True
        self.buttons['execute_day'].enabled = False
        self.update_stats()

    def on_next_day(self):
        """Handle Next Day button click."""
        print("\n[UI] Advancing to next day...")
        self.engine.advance_to_next_day()
        self.planned_routes = []
        self.package_status = {}

        # Reset button states
        self.buttons['plan_routes'].enabled = False
        self.buttons['execute_day'].enabled = False
        self.buttons['next_day'].enabled = False

        self.update_stats()

    def update_stats(self):
        """Update stat displays with current game state."""
        if not self.engine.game_state:
            return

        state = self.engine.game_state

        self.day_stat.set_value(str(state.current_day))

        balance_color = Colors.PROFIT_POSITIVE if state.balance > 0 else Colors.PROFIT_NEGATIVE
        self.balance_stat.set_value(f"${state.balance:,.0f}", balance_color)

        self.fleet_stat.set_value(f"{len(state.fleet)} vehicles")
        self.packages_stat.set_value(f"{len(state.packages_pending)} packages")

    def handle_events(self):
        """Handle all pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Quick test: plan routes on spacebar
                    if self.buttons['plan_routes'].enabled:
                        self.on_plan_routes()

            # Handle button clicks
            for button in self.buttons.values():
                button.handle_event(event)

            # Handle radio button clicks
            if event.type == pygame.MOUSEBUTTONDOWN:
                for i, radio in enumerate(self.agent_radios):
                    if radio.handle_event(event):
                        # Deselect others
                        for j, other_radio in enumerate(self.agent_radios):
                            if i != j:
                                other_radio.selected = False
                        self.selected_agent = radio.value
                        print(f"[UI] Selected agent: {self.selected_agent}")

            # Update radio button hover states
            if event.type == pygame.MOUSEMOTION:
                for radio in self.agent_radios:
                    radio.handle_event(event)

    def render(self):
        """Render everything."""
        # Clear screen
        self.screen.fill(Colors.BG_DARK)

        # Render title bar
        self.render_title_bar()

        # Render map
        self.render_map()

        # Render sidebar
        self.render_sidebar()

        # Render tooltip last (on top)
        self.tooltip.render(self.screen)

        # Update display
        pygame.display.flip()

    def render_title_bar(self):
        """Render the top title bar."""
        title_rect = pygame.Rect(0, 0, WINDOW_WIDTH, TITLE_BAR_HEIGHT)
        pygame.draw.rect(self.screen, Colors.TITLE_BG, title_rect)

        # Title
        font_large = pygame.font.Font(None, FontSizes.TITLE)
        title_text = font_large.render("DELIVERY FLEET MANAGER", True, Colors.TEXT_ACCENT)
        self.screen.blit(title_text, (20, 25))

        # Subtitle
        font_small = pygame.font.Font(None, FontSizes.SMALL)
        subtitle = font_small.render("Art of Programming - Route Optimization", True, Colors.TEXT_SECONDARY)
        self.screen.blit(subtitle, (20, 60))

        # Current status (right side)
        if self.engine.game_state:
            status_x = WINDOW_WIDTH - 350
            font_medium = pygame.font.Font(None, FontSizes.BODY)

            day_text = font_medium.render(f"Day {self.engine.game_state.current_day}", True, Colors.TEXT_PRIMARY)
            self.screen.blit(day_text, (status_x, 30))

            balance_color = Colors.PROFIT_POSITIVE if self.engine.game_state.balance >= 0 else Colors.PROFIT_NEGATIVE
            balance_text = font_medium.render(f"${self.engine.game_state.balance:,.2f}", True, balance_color)
            self.screen.blit(balance_text, (status_x, 55))

    def render_map(self):
        """Render the delivery map."""
        # Clear map surface
        self.map_surface.fill(Colors.MAP_BG)

        # Render map background and depot
        self.map_renderer.render_map_background()
        self.map_renderer.render_depot(pulse=True)

        # Render packages
        if self.engine.game_state and self.engine.game_state.packages_pending:
            for pkg in self.engine.game_state.packages_pending:
                status = self.package_status.get(pkg.id, "pending")
                self.map_renderer.render_package(pkg, status)

        # Render planned routes
        if self.planned_routes:
            for route in self.planned_routes:
                self.map_renderer.render_route(route, style="solid")

        # Render vehicles at depot (for now)
        if self.engine.game_state:
            for vehicle in self.engine.game_state.fleet:
                self.map_renderer.render_vehicle(vehicle)

        # Blit map surface to screen
        self.screen.blit(self.map_surface, (MAP_X, MAP_Y))

        # Draw border around map
        map_rect = pygame.Rect(MAP_X, MAP_Y, MAP_WIDTH, MAP_HEIGHT)
        pygame.draw.rect(self.screen, Colors.BORDER_LIGHT, map_rect, 2)

    def render_sidebar(self):
        """Render the sidebar with panels and controls."""
        # Render panels
        self.stats_panel.render(self.screen)
        self.agent_panel.render(self.screen)
        self.controls_panel.render(self.screen)

        # Render stats
        self.day_stat.render(self.screen)
        self.balance_stat.render(self.screen)
        self.fleet_stat.render(self.screen)
        self.packages_stat.render(self.screen)

        # Render agent radios
        for radio in self.agent_radios:
            radio.render(self.screen)

        # Render buttons
        for button in self.buttons.values():
            button.render(self.screen)

    def run(self):
        """Main game loop."""
        while self.running:
            # Handle events
            self.handle_events()

            # Update
            self.update_stats()

            # Render
            self.render()

            # Cap framerate
            self.clock.tick(FPS)

        pygame.quit()
        print("\nThank you for playing!")


def main():
    """Entry point."""
    try:
        app = DeliveryFleetApp()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()


if __name__ == "__main__":
    main()
