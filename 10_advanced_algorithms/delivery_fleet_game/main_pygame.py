#!/usr/bin/env python3
"""
Delivery Fleet Management System - Pygame GUI Version (PRODUCTION READY)

Complete, playable game with all features working.

Usage:
    python main_pygame.py
"""

import sys
import pygame
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import GameEngine
from src.agents import GreedyAgent, BacktrackingAgent, PruningBacktrackingAgent
from src.ui.constants import *
from src.ui.map_renderer import MapRenderer
from src.ui.components import Button, Panel, StatDisplay, RadioButton, Tooltip


class Modal:
    """Modal dialog for vehicle purchase and other actions."""

    def __init__(self, title: str, width: int = 500, height: int = 400):
        self.title = title
        self.width = width
        self.height = height
        self.visible = False
        self.buttons = []
        self.content_lines = []

        # Center position
        self.x = (WINDOW_WIDTH - width) // 2
        self.y = (WINDOW_HEIGHT - height) // 2
        self.rect = pygame.Rect(self.x, self.y, width, height)

    def show(self, content_lines: list, buttons: list):
        """Show modal with content and buttons."""
        self.visible = True
        self.content_lines = content_lines
        self.buttons = buttons

    def hide(self):
        """Hide modal."""
        self.visible = False
        self.buttons = []

    def handle_event(self, event):
        """Handle events for modal buttons."""
        if not self.visible:
            return None

        for button in self.buttons:
            if button.handle_event(event):
                return button.text  # Return which button was clicked
        return None

    def render(self, screen):
        """Render modal."""
        if not self.visible:
            return

        # Darken background
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))

        # Modal background
        pygame.draw.rect(screen, Colors.PANEL_BG, self.rect, border_radius=10)
        pygame.draw.rect(screen, Colors.BORDER_LIGHT, self.rect, 3, border_radius=10)

        # Title - Use SysFont for better rendering
        font_title = pygame.font.SysFont('arial', FontSizes.HEADING, bold=True)
        title_surf = font_title.render(self.title, True, Colors.TEXT_ACCENT)
        title_rect = title_surf.get_rect(center=(self.rect.centerx, self.y + 30))
        screen.blit(title_surf, title_rect)

        # Content - Use SysFont for better rendering
        font_body = pygame.font.SysFont('arial', FontSizes.BODY - 2)  # Slightly smaller for modal content
        y_offset = 70
        for line, color in self.content_lines:
            text_surf = font_body.render(line, True, color)
            text_rect = text_surf.get_rect(center=(self.rect.centerx, self.y + y_offset))
            screen.blit(text_surf, text_rect)
            y_offset += 25

        # Buttons
        for button in self.buttons:
            button.render(screen)


class DeliveryFleetApp:
    """Main Pygame application - PRODUCTION VERSION."""

    def __init__(self):
        """Initialize the application."""
        pygame.init()
        pygame.display.set_caption("Delivery Fleet Manager - Art of Programming")

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True

        # Surfaces
        self.map_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))

        # Initialize game engine
        data_dir = Path(__file__).parent / "data"
        self.engine = GameEngine(data_dir)
        self.map_renderer = MapRenderer(self.map_surface, self.engine.delivery_map)

        # Register agents
        self._register_agents()

        # UI state
        self.selected_agent = "greedy"
        self.tooltip = Tooltip()
        self.warning_message = ""
        self.warning_color = Colors.PROFIT_NEGATIVE

        # Modals
        self.vehicle_modal = Modal("Purchase Vehicle", 650, 650)  # Larger for detailed specs
        self.capacity_warning_modal = Modal("⚠️ Insufficient Capacity", 600, 400)
        self.marketing_modal = Modal("📈 Marketing & Package Rate", 650, 450)
        self.day_summary_modal = Modal("📦 Day Summary", 700, 500)

        # Create UI
        self._create_ui_components()

        # Game state
        self.planned_routes = []
        self.package_status = {}

        # Start new game
        self.engine.new_game()
        self.update_stats()

        print("✓ Delivery Fleet Manager Ready!")
        print("✓ Click 'Start Day' to begin")

    def _register_agents(self):
        """Register all routing agents."""
        self.engine.register_agent("greedy", GreedyAgent(self.engine.delivery_map))
        self.engine.register_agent("greedy_2opt", GreedyAgent(self.engine.delivery_map, use_2opt=True))
        self.engine.register_agent("backtracking", BacktrackingAgent(self.engine.delivery_map, max_packages=12))
        self.engine.register_agent("pruning_backtracking",
                                   PruningBacktrackingAgent(self.engine.delivery_map, max_packages=15))

    def _create_ui_components(self):
        """Create all UI components with FIXED layout."""

        # FIXED LAYOUT - Everything fits within 800px height
        SIDEBAR_START = 100  # Below title bar

        # Panels - Adjusted heights to fit
        self.stats_panel = Panel(SIDEBAR_X + 10, SIDEBAR_START, SIDEBAR_WIDTH - 20, 170, "GAME STATUS")
        self.agent_panel = Panel(SIDEBAR_X + 10, SIDEBAR_START + 180, SIDEBAR_WIDTH - 20, 240, "AGENTS")
        self.controls_panel = Panel(SIDEBAR_X + 10, SIDEBAR_START + 430, SIDEBAR_WIDTH - 20, 260, "CONTROLS")

        # Warning message area
        self.warning_rect = pygame.Rect(SIDEBAR_X + 10, SIDEBAR_START + 700, SIDEBAR_WIDTH - 20, 40)

        # Stats - Compact layout
        stat_x = SIDEBAR_X + 25
        self.day_stat = StatDisplay(stat_x, SIDEBAR_START + 40, "Day:", "1")
        self.balance_stat = StatDisplay(stat_x + 140, SIDEBAR_START + 40, "Balance:", "$100K")
        self.fleet_stat = StatDisplay(stat_x, SIDEBAR_START + 100, "Fleet:", "2 veh")
        self.packages_stat = StatDisplay(stat_x + 140, SIDEBAR_START + 100, "Pending:", "0")
        self.capacity_stat = StatDisplay(stat_x + 280, SIDEBAR_START + 40, "Capacity:", "0/0")

        # Planned route metrics (visible when routes are planned)
        self.planned_cost_stat = StatDisplay(stat_x, SIDEBAR_START + 135, "Cost:", "$0")
        self.planned_revenue_stat = StatDisplay(stat_x + 140, SIDEBAR_START + 135, "Revenue:", "$0")
        self.planned_profit_stat = StatDisplay(stat_x + 280, SIDEBAR_START + 135, "Profit:", "$0")

        # Store planned metrics
        self.planned_metrics = None

        # Agent radio buttons - Compact
        radio_x = SIDEBAR_X + 35
        radio_y = SIDEBAR_START + 220
        self.agent_radios = [
            RadioButton(radio_x, radio_y, "Greedy", "agent", "greedy"),
            RadioButton(radio_x, radio_y + 35, "Greedy+2opt", "agent", "greedy_2opt"),
            RadioButton(radio_x, radio_y + 70, "Backtrack", "agent", "backtracking"),
            RadioButton(radio_x, radio_y + 105, "Pruning BT", "agent", "pruning_backtracking"),
        ]
        self.agent_radios[0].selected = True

        # Control buttons - Optimized spacing to fit all buttons
        btn_x = SIDEBAR_X + 25
        btn_y = SIDEBAR_START + 470
        btn_width = SIDEBAR_WIDTH - 50
        btn_small_width = (btn_width - 10) // 2
        btn_height = 32  # Reduced from 35 to fit everything
        btn_spacing = 36  # Tight spacing

        self.buttons = {
            'start_day': Button(btn_x, btn_y, btn_width, btn_height, "📦 Start Day", self.on_start_day),
            'buy_vehicle': Button(btn_x, btn_y + btn_spacing, btn_width, btn_height, "🚚 Buy Vehicle", self.on_buy_vehicle),
            'plan_routes': Button(btn_x, btn_y + btn_spacing * 2, btn_small_width, btn_height, "🧠 Plan", self.on_plan_routes),
            'clear': Button(btn_x + btn_small_width + 10, btn_y + btn_spacing * 2, btn_small_width, btn_height, "🔄 Clear", self.on_clear_routes),
            'execute': Button(btn_x, btn_y + btn_spacing * 3, btn_small_width, btn_height, "▶️ Execute", self.on_execute_day),
            'next_day': Button(btn_x + btn_small_width + 10, btn_y + btn_spacing * 3, btn_small_width, btn_height, "⏭️ Next", self.on_next_day),
            'marketing': Button(btn_x, btn_y + btn_spacing * 4, btn_width, btn_height, "📈 Marketing", self.on_show_marketing),
            'save': Button(btn_x, btn_y + btn_spacing * 5, btn_small_width, btn_height, "💾 Save", self.on_save),
            'stats': Button(btn_x + btn_small_width + 10, btn_y + btn_spacing * 5, btn_small_width, btn_height, "📊 Stats", self.on_stats),
        }

        # Set initial states
        self.buttons['plan_routes'].enabled = False
        self.buttons['clear'].enabled = False
        self.buttons['execute'].enabled = False
        self.buttons['next_day'].enabled = False

    # ==================== EVENT HANDLERS ====================

    def on_start_day(self):
        """Start a new day."""
        print("\n[UI] Starting day...")
        self.engine.start_day()

        if not self.engine.game_state.packages_pending:
            self.show_warning("No packages for this day!", Colors.TEXT_ACCENT)
            return

        self.package_status = {pkg.id: "pending" for pkg in self.engine.game_state.packages_pending}

        # Check capacity
        total_volume = sum(pkg.volume_m3 for pkg in self.engine.game_state.packages_pending)
        fleet_capacity = sum(v.vehicle_type.capacity_m3 for v in self.engine.game_state.fleet)

        # Show day summary
        self.show_day_summary(total_volume, fleet_capacity)

        if total_volume > fleet_capacity:
            # Will show capacity warning after closing day summary
            pass
        else:
            self.buttons['plan_routes'].enabled = True
            self.show_warning("", Colors.TEXT_PRIMARY)

        self.update_stats()

    def show_day_summary(self, total_volume: float, fleet_capacity: float):
        """Show day start summary with package and fleet information."""
        state = self.engine.game_state
        packages = state.packages_pending

        # Count package types
        small_pkgs = sum(1 for p in packages if p.volume_m3 < 2.5)
        medium_pkgs = sum(1 for p in packages if 2.5 <= p.volume_m3 < 4.0)
        large_pkgs = sum(1 for p in packages if p.volume_m3 >= 4.0)
        priority_pkgs = sum(1 for p in packages if p.priority >= 3)

        # Calculate potential revenue
        potential_revenue = sum(p.payment for p in packages)

        # Fleet breakdown
        fleet_by_type = {}
        for v in state.fleet:
            vtype = v.vehicle_type.name
            if vtype not in fleet_by_type:
                fleet_by_type[vtype] = 0
            fleet_by_type[vtype] += 1

        # Capacity status
        capacity_pct = (total_volume / fleet_capacity * 100) if fleet_capacity > 0 else 0
        capacity_color = Colors.PROFIT_POSITIVE if capacity_pct <= 100 else Colors.PROFIT_NEGATIVE

        content = [
            (f"═══ DAY {state.current_day} START ═══", Colors.TEXT_ACCENT),
            ("", Colors.TEXT_PRIMARY),
            ("📦 PACKAGES TO DELIVER", Colors.TEXT_ACCENT),
            (f"   Total: {len(packages)} packages ({total_volume:.1f}m³)", Colors.TEXT_PRIMARY),
            (f"   Small: {small_pkgs} | Medium: {medium_pkgs} | Large: {large_pkgs}", Colors.TEXT_SECONDARY),
            (f"   High Priority: {priority_pkgs}", Colors.TEXT_SECONDARY),
            (f"   Potential Revenue: ${potential_revenue:.0f}", Colors.PROFIT_POSITIVE),
            ("", Colors.TEXT_PRIMARY),
            ("🚚 FLEET STATUS", Colors.TEXT_ACCENT),
            (f"   Total Capacity: {fleet_capacity:.1f}m³", Colors.TEXT_PRIMARY),
        ]

        # Add fleet breakdown
        for vtype, count in fleet_by_type.items():
            content.append((f"   {vtype}: {count}x", Colors.TEXT_SECONDARY))

        content.extend([
            ("", Colors.TEXT_PRIMARY),
            (f"📊 CAPACITY USAGE: {capacity_pct:.0f}%", capacity_color),
            ("", Colors.TEXT_PRIMARY),
            ("💡 Hover over packages on map for details!", Colors.TEXT_ACCENT),
            ("", Colors.TEXT_PRIMARY),  # Extra spacing before buttons
            ("", Colors.TEXT_PRIMARY),  # Extra spacing before buttons
        ])

        # Create buttons - positioned lower to avoid overlap
        modal_btn_x = self.day_summary_modal.x + 180
        modal_btn_y = self.day_summary_modal.y + 440  # Increased from 420

        if total_volume > fleet_capacity:
            # Need more capacity
            buttons = [
                Button(modal_btn_x, modal_btn_y, 250, 40, "⚠️ Buy Vehicle (Shortage!)",
                       lambda: self.close_day_summary_and_buy(total_volume, fleet_capacity)),
                Button(modal_btn_x + 260, modal_btn_y, 150, 40, "Continue",
                       lambda: self.close_day_summary_with_warning(total_volume, fleet_capacity)),
            ]
        else:
            buttons = [
                Button(modal_btn_x + 120, modal_btn_y, 250, 40, "Start Planning Routes ✓",
                       lambda: self.day_summary_modal.hide()),
            ]

        self.day_summary_modal.show(content, buttons)

    def close_day_summary_and_buy(self, needed: float, available: float):
        """Close day summary and open vehicle purchase."""
        self.day_summary_modal.hide()
        self.on_buy_vehicle()

    def close_day_summary_with_warning(self, needed: float, available: float):
        """Close day summary and show capacity warning."""
        self.day_summary_modal.hide()
        self.show_capacity_warning(needed, available)

    def show_capacity_warning(self, needed: float, available: float):
        """Show warning when capacity is insufficient."""
        deficit = needed - available
        needed_capacity = deficit

        # Suggest vehicle to buy
        suggestion = ""
        for vtype_name, vtype in self.engine.vehicle_types.items():
            if vtype.capacity_m3 >= deficit:
                if vtype.purchase_price <= self.engine.game_state.balance:
                    suggestion = f"Buy {vtype.name} ({vtype.capacity_m3}m³) for ${vtype.purchase_price:,}"
                    break

        content = [
            ("⚠️ CAPACITY PROBLEM", Colors.PROFIT_NEGATIVE),
            ("", Colors.TEXT_PRIMARY),
            (f"Total packages: {needed:.1f}m³", Colors.TEXT_PRIMARY),
            (f"Fleet capacity: {available:.1f}m³", Colors.TEXT_PRIMARY),
            (f"Shortage: {deficit:.1f}m³", Colors.PROFIT_NEGATIVE),
            ("", Colors.TEXT_PRIMARY),
            ("You need more vehicles!", Colors.TEXT_ACCENT),
            (suggestion if suggestion else "Not enough balance!", Colors.TEXT_SECONDARY),
        ]

        # Create buttons
        modal_btn_y = self.capacity_warning_modal.y + 320
        modal_btn_x = self.capacity_warning_modal.x + 50
        modal_btn_width = 200

        buttons = [
            Button(modal_btn_x, modal_btn_y, modal_btn_width, 40, "Buy Vehicle",
                   lambda: self.close_modal_and_buy()),
            Button(modal_btn_x + 220, modal_btn_y, modal_btn_width, 40, "Skip Day",
                   lambda: self.close_modal_and_skip()),
        ]

        self.capacity_warning_modal.show(content, buttons)

    def close_modal_and_buy(self):
        """Close modal and open buy vehicle."""
        self.capacity_warning_modal.hide()
        self.on_buy_vehicle()

    def close_modal_and_skip(self):
        """Close modal and advance day."""
        self.capacity_warning_modal.hide()
        self.on_next_day()

    def on_buy_vehicle(self):
        """Show vehicle purchase modal."""
        content = [
            ("🚚 VEHICLE PURCHASE", Colors.TEXT_ACCENT),
            ("", Colors.TEXT_PRIMARY),
            (f"Your Balance: ${self.engine.game_state.balance:,.0f}", Colors.PROFIT_POSITIVE),
            ("", Colors.TEXT_PRIMARY),
        ]

        # List vehicles with detailed specs
        y_offset = 100
        buttons = []

        for vtype_name, vtype in self.engine.vehicle_types.items():
            can_afford = vtype.purchase_price <= self.engine.game_state.balance
            name_color = Colors.TEXT_ACCENT if can_afford else Colors.TEXT_SECONDARY
            spec_color = Colors.TEXT_PRIMARY if can_afford else Colors.TEXT_SECONDARY

            # Vehicle name with affordability indicator
            affordability = "✓ AFFORDABLE" if can_afford else "✗ INSUFFICIENT FUNDS"
            afford_color = Colors.PROFIT_POSITIVE if can_afford else Colors.PROFIT_NEGATIVE

            content.append((f"━━━ {vtype.name.upper()} ━━━", name_color))
            content.append((f"   Capacity: {vtype.capacity_m3:.0f} m³", spec_color))
            content.append((f"   Purchase Price: ${vtype.purchase_price:,}", spec_color))
            content.append((f"   Operating Cost: ${vtype.cost_per_km:.2f}/km", spec_color))
            content.append((f"   Range: {vtype.max_range_km:.0f} km", spec_color))
            content.append((f"   {affordability}", afford_color))
            content.append(("", Colors.TEXT_PRIMARY))

            # Create button
            btn_x = self.vehicle_modal.x + 150
            btn_y = self.vehicle_modal.y + y_offset
            btn_width = 300

            btn = Button(btn_x, btn_y, btn_width, 38,
                        f"Purchase {vtype.name} - ${vtype.purchase_price:,}",
                        lambda vt=vtype_name: self.purchase_vehicle(vt))
            btn.enabled = can_afford
            buttons.append(btn)

            y_offset += 150  # More spacing between vehicles

        # Cancel button at bottom
        cancel_btn = Button(self.vehicle_modal.x + 200, self.vehicle_modal.y + self.vehicle_modal.height - 60,
                          200, 40, "Cancel", lambda: self.vehicle_modal.hide())
        buttons.append(cancel_btn)

        self.vehicle_modal.show(content, buttons)

    def purchase_vehicle(self, vehicle_type_name: str):
        """Purchase a vehicle."""
        vehicle_type = self.engine.vehicle_types[vehicle_type_name]
        vehicle_id = f"veh_{len(self.engine.game_state.fleet) + 1:03d}"

        if self.engine.game_state.purchase_vehicle(vehicle_type, vehicle_id):
            print(f"✓ Purchased {vehicle_type.name}")
            self.show_warning(f"Purchased {vehicle_type.name}!", Colors.PROFIT_POSITIVE)
            self.update_stats()
            self.vehicle_modal.hide()

            # Re-check if we can now plan routes
            if self.engine.game_state.packages_pending:
                total_volume = sum(pkg.volume_m3 for pkg in self.engine.game_state.packages_pending)
                fleet_capacity = sum(v.vehicle_type.capacity_m3 for v in self.engine.game_state.fleet)
                if total_volume <= fleet_capacity:
                    self.buttons['plan_routes'].enabled = True
        else:
            self.show_warning("Not enough funds!", Colors.PROFIT_NEGATIVE)

    def on_plan_routes(self):
        """Plan routes with selected agent."""
        print(f"\n[UI] Planning with {self.selected_agent}...")
        metrics = self.engine.test_agent(self.selected_agent)

        if metrics and metrics.get('routes'):
            self.planned_routes = metrics['routes']
            self.planned_metrics = metrics
            self.engine.apply_agent_solution(self.selected_agent)
            self.buttons['execute'].enabled = True
            self.buttons['clear'].enabled = True

            # Update planned metrics display
            cost = metrics['total_cost']
            revenue = metrics['total_revenue']
            profit = metrics['total_profit']

            profit_color = Colors.PROFIT_POSITIVE if profit > 0 else Colors.PROFIT_NEGATIVE
            self.planned_cost_stat.set_value(f"${cost:.0f}", Colors.PROFIT_NEGATIVE)
            self.planned_revenue_stat.set_value(f"${revenue:.0f}", Colors.PROFIT_POSITIVE)
            self.planned_profit_stat.set_value(f"${profit:.0f}", profit_color)

            self.show_warning(f"Routes planned! Profit: ${profit:.2f}", Colors.PROFIT_POSITIVE)
        else:
            self.show_warning("No valid routes found!", Colors.PROFIT_NEGATIVE)

    def on_clear_routes(self):
        """Clear planned routes and reset UI."""
        print("\n[UI] Clearing routes...")
        self.planned_routes = []
        self.planned_metrics = None
        self.engine.game_state.set_routes([])
        self.buttons['execute'].enabled = False
        self.buttons['clear'].enabled = False

        # Clear planned metrics display
        self.planned_cost_stat.set_value("$0", Colors.TEXT_SECONDARY)
        self.planned_revenue_stat.set_value("$0", Colors.TEXT_SECONDARY)
        self.planned_profit_stat.set_value("$0", Colors.TEXT_SECONDARY)

        self.show_warning("Routes cleared", Colors.TEXT_ACCENT)

    def on_execute_day(self):
        """Execute the planned routes."""
        print("\n[UI] Executing day...")
        self.engine.execute_day(self.selected_agent)

        for pkg in self.engine.game_state.packages_delivered:
            if pkg.id in self.package_status:
                self.package_status[pkg.id] = "delivered"

        self.buttons['next_day'].enabled = True
        self.buttons['execute'].enabled = False
        self.buttons['clear'].enabled = False
        self.buttons['plan_routes'].enabled = False

        # Clear planned metrics display
        self.planned_metrics = None
        self.planned_cost_stat.set_value("$0", Colors.TEXT_SECONDARY)
        self.planned_revenue_stat.set_value("$0", Colors.TEXT_SECONDARY)
        self.planned_profit_stat.set_value("$0", Colors.TEXT_SECONDARY)

        # Show result
        last_day = self.engine.game_state.get_last_day_summary()
        if last_day:
            msg = f"Day complete! Profit: ${last_day.profit:+.2f}"
            color = Colors.PROFIT_POSITIVE if last_day.profit > 0 else Colors.PROFIT_NEGATIVE
            self.show_warning(msg, color)

        self.update_stats()

    def on_next_day(self):
        """Advance to next day."""
        self.engine.advance_to_next_day()
        self.planned_routes = []
        self.package_status = {}
        self.buttons['plan_routes'].enabled = False
        self.buttons['execute'].enabled = False
        self.buttons['next_day'].enabled = False
        self.show_warning("", Colors.TEXT_PRIMARY)
        self.update_stats()

    def on_save(self):
        """Save game."""
        self.engine.save_game()
        self.show_warning("Game saved!", Colors.TEXT_ACCENT)

    def on_stats(self):
        """Show statistics."""
        stats = self.engine.game_state.get_statistics()
        print("\n" + "="*50)
        print("GAME STATISTICS")
        print("="*50)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("="*50)
        self.show_warning("Stats printed to console", Colors.TEXT_ACCENT)

    def on_show_marketing(self):
        """Show marketing upgrade modal."""
        marketing_info = self.engine.game_state.get_marketing_info()

        content = [
            ("📈 MARKETING & PACKAGE RATE", Colors.TEXT_ACCENT),
            ("", Colors.TEXT_PRIMARY),
            (f"Current Level: {marketing_info['level']}/5", Colors.TEXT_PRIMARY),
            (f"Daily Package Volume: {marketing_info['current_volume']:.1f}m³", Colors.PROFIT_POSITIVE),
            ("", Colors.TEXT_PRIMARY),
        ]

        if not marketing_info['is_max_level']:
            next_level = marketing_info['level'] + 1
            next_volume = marketing_info['next_level_volume']
            upgrade_cost = marketing_info['upgrade_cost']

            content.extend([
                (f"Next Level ({next_level}): {next_volume:.1f}m³/day", Colors.TEXT_SECONDARY),
                (f"Upgrade Cost: ${upgrade_cost:,}", Colors.TEXT_SECONDARY),
                ("", Colors.TEXT_PRIMARY),
                ("💡 Higher marketing = More packages!", Colors.TEXT_ACCENT),
                ("   Grow your fleet to handle increased volume", Colors.TEXT_SECONDARY),
            ])
        else:
            content.extend([
                ("🎉 MAX LEVEL REACHED!", Colors.PROFIT_POSITIVE),
                ("You have maximum marketing coverage!", Colors.TEXT_SECONDARY),
            ])

        # Create buttons
        buttons = []
        modal_btn_x = self.marketing_modal.x + 100
        modal_btn_y = self.marketing_modal.y + 360

        if not marketing_info['is_max_level']:
            upgrade_btn = Button(
                modal_btn_x, modal_btn_y, 200, 40,
                f"Upgrade (${marketing_info['upgrade_cost']:,})",
                self.on_upgrade_marketing
            )
            upgrade_btn.enabled = marketing_info['can_afford']
            buttons.append(upgrade_btn)

            close_btn = Button(
                modal_btn_x + 220, modal_btn_y, 200, 40,
                "Close", lambda: self.marketing_modal.hide()
            )
            buttons.append(close_btn)
        else:
            close_btn = Button(
                modal_btn_x + 110, modal_btn_y, 200, 40,
                "Close", lambda: self.marketing_modal.hide()
            )
            buttons.append(close_btn)

        self.marketing_modal.show(content, buttons)

    def on_upgrade_marketing(self):
        """Upgrade marketing level."""
        old_info = self.engine.game_state.get_marketing_info()

        if self.engine.game_state.upgrade_marketing():
            new_info = self.engine.game_state.get_marketing_info()
            print(f"✓ Marketing upgraded to level {new_info['level']}")
            self.show_warning(
                f"Marketing upgraded! Now {new_info['current_volume']:.1f}m³/day",
                Colors.PROFIT_POSITIVE
            )
            self.update_stats()
            self.marketing_modal.hide()
        else:
            self.show_warning("Cannot upgrade marketing!", Colors.PROFIT_NEGATIVE)

    def show_warning(self, message: str, color):
        """Show warning message."""
        self.warning_message = message
        self.warning_color = color

    def update_stats(self):
        """Update all stat displays."""
        if not self.engine.game_state:
            return

        state = self.engine.game_state

        self.day_stat.set_value(str(state.current_day))

        balance_color = Colors.PROFIT_POSITIVE if state.balance > 0 else Colors.PROFIT_NEGATIVE
        if state.balance >= 1000:
            balance_str = f"${state.balance/1000:.1f}K"
        else:
            balance_str = f"${state.balance:.0f}"
        self.balance_stat.set_value(balance_str, balance_color)

        self.fleet_stat.set_value(f"{len(state.fleet)} veh")
        self.packages_stat.set_value(f"{len(state.packages_pending)}")

        # Capacity
        total_pkg_volume = sum(pkg.volume_m3 for pkg in state.packages_pending) if state.packages_pending else 0
        fleet_capacity = sum(v.vehicle_type.capacity_m3 for v in state.fleet)
        capacity_color = Colors.PROFIT_POSITIVE if total_pkg_volume <= fleet_capacity else Colors.PROFIT_NEGATIVE
        self.capacity_stat.set_value(f"{total_pkg_volume:.0f}/{fleet_capacity:.0f}", capacity_color)

    def update_hover_tooltip(self, mouse_pos):
        """Update tooltip based on mouse position."""
        if not self.engine.game_state:
            self.tooltip.hide()
            return

        # Check packages
        if self.engine.game_state.packages_pending:
            pkg = self.map_renderer.get_package_at_mouse(mouse_pos, self.engine.game_state.packages_pending)
            if pkg:
                status = self.package_status.get(pkg.id, "pending")
                status_text = "✓ DELIVERED" if status == "delivered" else "📦 PENDING"
                tooltip_text = f"{status_text}\n{pkg.id}: {pkg.description or 'Package'}\nVolume: {pkg.volume_m3}m³\nPayment: ${pkg.payment}\nPriority: {pkg.priority}"
                self.tooltip.show(tooltip_text, (mouse_pos[0] + 15, mouse_pos[1] + 15))
                return

        # Check vehicles
        if self.engine.game_state.fleet:
            veh = self.map_renderer.get_vehicle_at_mouse(mouse_pos, self.engine.game_state.fleet)
            if veh:
                tooltip_text = f"🚚 {veh.vehicle_type.name}\n{veh.id}\nCapacity: {veh.vehicle_type.capacity_m3}m³\nCost: ${veh.vehicle_type.cost_per_km}/km\nRange: {veh.vehicle_type.max_range_km}km"
                self.tooltip.show(tooltip_text, (mouse_pos[0] + 15, mouse_pos[1] + 15))
                return

        # No hover
        self.tooltip.hide()

    # ==================== MAIN LOOP ====================

    def handle_events(self):
        """Handle all events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # Close modals first, then quit
                    if self.vehicle_modal.visible:
                        self.vehicle_modal.hide()
                    elif self.capacity_warning_modal.visible:
                        self.capacity_warning_modal.hide()
                    elif self.marketing_modal.visible:
                        self.marketing_modal.hide()
                    elif self.day_summary_modal.visible:
                        self.day_summary_modal.hide()
                    else:
                        self.running = False

            # Modal events (priority)
            if self.day_summary_modal.visible:
                self.day_summary_modal.handle_event(event)
                continue  # Don't process other events

            if self.vehicle_modal.visible:
                self.vehicle_modal.handle_event(event)
                continue  # Don't process other events

            if self.capacity_warning_modal.visible:
                self.capacity_warning_modal.handle_event(event)
                continue

            if self.marketing_modal.visible:
                self.marketing_modal.handle_event(event)
                continue

            # Regular button events
            for button in self.buttons.values():
                button.handle_event(event)

            # Radio buttons
            if event.type == pygame.MOUSEBUTTONDOWN:
                for i, radio in enumerate(self.agent_radios):
                    if radio.handle_event(event):
                        for j, other in enumerate(self.agent_radios):
                            if i != j:
                                other.selected = False
                        self.selected_agent = radio.value

            if event.type == pygame.MOUSEMOTION:
                for radio in self.agent_radios:
                    radio.handle_event(event)

                # Check for hover over packages/vehicles
                self.update_hover_tooltip(event.pos)

    def render(self):
        """Render everything."""
        self.screen.fill(Colors.BG_DARK)

        self.render_title_bar()
        self.render_map()
        self.render_sidebar()

        # Modals on top
        self.day_summary_modal.render(self.screen)
        self.vehicle_modal.render(self.screen)
        self.capacity_warning_modal.render(self.screen)
        self.marketing_modal.render(self.screen)

        self.tooltip.render(self.screen)

        pygame.display.flip()

    def render_title_bar(self):
        """Render title bar."""
        title_rect = pygame.Rect(0, 0, WINDOW_WIDTH, TITLE_BAR_HEIGHT)
        pygame.draw.rect(self.screen, Colors.TITLE_BG, title_rect)

        # Use SysFont for better anti-aliasing
        font_large = pygame.font.SysFont('arial', FontSizes.TITLE, bold=True)
        title = font_large.render("DELIVERY FLEET MANAGER", True, Colors.TEXT_ACCENT)
        self.screen.blit(title, (20, 20))

        font_small = pygame.font.SysFont('arial', FontSizes.SMALL)
        subtitle = font_small.render("Art of Programming - Route Optimization", True, Colors.TEXT_SECONDARY)
        self.screen.blit(subtitle, (20, 58))

        # Status - Improved rendering with panel background
        if self.engine.game_state:
            status_x = WINDOW_WIDTH - 280
            status_y = 15

            # Draw background panel for status
            status_panel = pygame.Rect(status_x - 15, status_y - 5, 270, 70)
            pygame.draw.rect(self.screen, Colors.PANEL_BG, status_panel, border_radius=8)
            pygame.draw.rect(self.screen, Colors.BORDER_LIGHT, status_panel, 2, border_radius=8)

            # Day label and value
            font_label = pygame.font.SysFont('arial', 14)
            font_value = pygame.font.SysFont('arial', 24, bold=True)

            day_label = font_label.render("Day", True, Colors.TEXT_SECONDARY)
            self.screen.blit(day_label, (status_x, status_y))

            day_value = font_value.render(str(self.engine.game_state.current_day), True, Colors.TEXT_ACCENT)
            self.screen.blit(day_value, (status_x, status_y + 18))

            # Balance label and value
            bal_x = status_x + 120
            bal_label = font_label.render("Balance", True, Colors.TEXT_SECONDARY)
            self.screen.blit(bal_label, (bal_x, status_y))

            bal_color = Colors.PROFIT_POSITIVE if self.engine.game_state.balance >= 0 else Colors.PROFIT_NEGATIVE
            bal_text = f"${self.engine.game_state.balance:,.0f}"
            bal_value = font_value.render(bal_text, True, bal_color)
            self.screen.blit(bal_value, (bal_x, status_y + 18))

    def render_map(self):
        """Render the map."""
        self.map_surface.fill(Colors.MAP_BG)
        self.map_renderer.render_map_background()
        self.map_renderer.render_depot(pulse=True)

        if self.engine.game_state and self.engine.game_state.packages_pending:
            for pkg in self.engine.game_state.packages_pending:
                status = self.package_status.get(pkg.id, "pending")
                self.map_renderer.render_package(pkg, status)

        if self.planned_routes:
            for i, route in enumerate(self.planned_routes):
                # Assign distinct color to each route
                route_color = Colors.ROUTE_COLORS[i % len(Colors.ROUTE_COLORS)]
                self.map_renderer.render_route(route, color=route_color, style="solid")

        if self.engine.game_state:
            for vehicle in self.engine.game_state.fleet:
                self.map_renderer.render_vehicle(vehicle)

        self.screen.blit(self.map_surface, (MAP_X, MAP_Y))
        map_rect = pygame.Rect(MAP_X, MAP_Y, MAP_WIDTH, MAP_HEIGHT)
        pygame.draw.rect(self.screen, Colors.BORDER_LIGHT, map_rect, 2)

        # Render legend below map
        self.render_map_legend()

    def render_map_legend(self):
        """Render horizontal legend below the map."""
        legend_y = MAP_Y + MAP_HEIGHT + 10
        legend_width = MAP_WIDTH
        legend_height = 80
        legend_x = MAP_X

        # Background
        legend_rect = pygame.Rect(legend_x, legend_y, legend_width, legend_height)
        pygame.draw.rect(self.screen, Colors.PANEL_BG, legend_rect, border_radius=5)
        pygame.draw.rect(self.screen, Colors.BORDER_LIGHT, legend_rect, 2, border_radius=5)

        # Title - Use SysFont for better rendering
        font_title = pygame.font.SysFont('arial', FontSizes.BODY, bold=True)
        title = font_title.render("MAP LEGEND", True, Colors.TEXT_ACCENT)
        self.screen.blit(title, (legend_x + 10, legend_y + 8))

        # Legend items in 2 rows, 3 columns - Use SysFont
        font_small = pygame.font.SysFont('arial', FontSizes.SMALL)

        # Column 1
        x_col1 = legend_x + 15
        y_row1 = legend_y + 35
        y_row2 = legend_y + 55

        # Depot
        pygame.draw.circle(self.screen, Colors.DEPOT, (x_col1, y_row1), 6)
        text = font_small.render("Depot", True, Colors.TEXT_PRIMARY)
        self.screen.blit(text, (x_col1 + 12, y_row1 - 6))

        # Pending
        pygame.draw.circle(self.screen, Colors.PACKAGE_PENDING, (x_col1, y_row2), 5)
        text = font_small.render("Pending", True, Colors.TEXT_PRIMARY)
        self.screen.blit(text, (x_col1 + 12, y_row2 - 6))

        # Column 2
        x_col2 = legend_x + 150

        # Delivered
        pygame.draw.circle(self.screen, Colors.PACKAGE_DELIVERED, (x_col2, y_row1), 5)
        text = font_small.render("Delivered", True, Colors.TEXT_PRIMARY)
        self.screen.blit(text, (x_col2 + 12, y_row1 - 6))

        # High priority
        pygame.draw.circle(self.screen, Colors.PACKAGE_PRIORITY_HIGH, (x_col2, y_row2), 5)
        text = font_small.render("Priority 3+", True, Colors.TEXT_PRIMARY)
        self.screen.blit(text, (x_col2 + 12, y_row2 - 6))

        # Column 3
        x_col3 = legend_x + 300

        # Route
        pygame.draw.line(self.screen, Colors.ROUTE_COLORS[0],
                        (x_col3, y_row1), (x_col3 + 20, y_row1), 3)
        # Arrow
        pygame.draw.polygon(self.screen, Colors.ROUTE_COLORS[0], [
            (x_col3 + 20, y_row1),
            (x_col3 + 16, y_row1 - 3),
            (x_col3 + 16, y_row1 + 3)
        ])
        text = font_small.render("Route", True, Colors.TEXT_PRIMARY)
        self.screen.blit(text, (x_col3 + 28, y_row1 - 6))

        # Vehicle
        veh_rect = pygame.Rect(x_col3 + 2, y_row2 - 4, 12, 8)
        pygame.draw.rect(self.screen, Colors.VEHICLE_ACTIVE, veh_rect, border_radius=1)
        text = font_small.render("Vehicle", True, Colors.TEXT_PRIMARY)
        self.screen.blit(text, (x_col3 + 28, y_row2 - 6))

        # Column 4 - Hint
        x_col4 = legend_x + 480
        hint_font = pygame.font.Font(None, FontSizes.SMALL)
        hint1 = hint_font.render("💡 Hover over packages", True, Colors.TEXT_ACCENT)
        hint2 = hint_font.render("   and vehicles for info!", True, Colors.TEXT_ACCENT)
        self.screen.blit(hint1, (x_col4, y_row1 - 6))
        self.screen.blit(hint2, (x_col4, y_row2 - 6))

    def render_sidebar(self):
        """Render sidebar."""
        # Panels
        self.stats_panel.render(self.screen)
        self.agent_panel.render(self.screen)
        self.controls_panel.render(self.screen)

        # Stats
        self.day_stat.render(self.screen)
        self.balance_stat.render(self.screen)
        self.fleet_stat.render(self.screen)
        self.packages_stat.render(self.screen)
        self.capacity_stat.render(self.screen)

        # Planned route metrics (only show when routes are planned)
        if self.planned_metrics:
            self.planned_cost_stat.render(self.screen)
            self.planned_revenue_stat.render(self.screen)
            self.planned_profit_stat.render(self.screen)

        # Radios
        for radio in self.agent_radios:
            radio.render(self.screen)

        # Buttons
        for button in self.buttons.values():
            button.render(self.screen)

        # Warning message
        if self.warning_message:
            font = pygame.font.Font(None, FontSizes.SMALL)
            # Word wrap
            words = self.warning_message.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if font.size(test_line)[0] < self.warning_rect.width - 20:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)

            y_offset = self.warning_rect.y + 10
            for line in lines:
                text = font.render(line, True, self.warning_color)
                text_rect = text.get_rect(center=(self.warning_rect.centerx, y_offset))
                self.screen.blit(text, text_rect)
                y_offset += 20

    def run(self):
        """Main game loop."""
        while self.running:
            self.handle_events()
            self.render()
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
