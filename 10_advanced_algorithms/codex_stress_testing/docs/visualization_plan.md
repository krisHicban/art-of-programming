# Visualization Plan – Delivery Fleet Management System

## Goals
- Provide a clean, educational, and interactive view of routes, fleet utilization, and financial metrics.
- Maintain phase-driven clarity: Planning, Execution, End-of-Day summaries.
- Surface algorithm comparisons visually (Greedy vs upcoming agents) to reinforce learning outcomes.

## Stack & Libraries
- **Pygame** for 2D rendering, input handling, and lightweight animation.
- **PyTMX (optional)** if we later adopt tiled maps; initial release uses procedural grid rendering.
- **PyTweening** (optional) for smooth interpolation of vehicle movements during execution replay.
- **Tabulate / Rich** remain for console output; Pygame UI mirrors key data.
- Keep third-party additions minimal to avoid install friction.

## Architectural Outline
```
ui/
├── app.py            # Pygame bootstrap & main loop
├── controllers.py    # Mediates between GameState snapshots and views
├── theme.py          # Colors, typography, spacing constants
├── views/
│   ├── map_view.py   # Renders grid, depot, package destinations, vehicle routes
│   ├── hud_view.py   # KPI panels: balance, daily stats, agent comparisons
│   └── timeline.py   # Event timeline using GameState.events
└── widgets/
    ├── button.py
    └── panel.py
```

## UX Flow
1. **Dashboard Screen**
   - Left: Map with depot-centered Cartesian grid, packages as icons, animated vehicle markers along routes.
   - Right: Panels for balance/trends, fleet utilization, agent metrics comparison.
   - Bottom: Timeline scrubber showing events (day start, assignments, validations, completion) with active-event highlight.
2. **Playback Controls**
   - Play/pause to animate execution using snapshots.
   - Dropdown to select day (populated from exported snapshots).
3. **Agent Comparison Overlay**
   - Toggle to show side-by-side stats for latest agent runs (distance, cost, profit, unassigned packages).
4. **Design Language**
   - Light background with accent colors per vehicle category.
   - Rounded panels, subtle shadows; consistent typography via custom font loading.

## Data Flow
- Engine exports `savegames/day_##.json` and `snapshots/day_##.json`.
- UI loads snapshots via `utils.exporter.build_state_snapshot` or directly from file.
- Event log drives timeline and tooltip content.
- Agent history populates comparison charts.

## Next Steps
1. Implement `ui.app` bootstrap reading latest snapshot.
2. Build `MapView` with scalable grid and icon rendering (Pygame primitives initially).
3. Create HUD panels for balance and agent metrics.
4. Hook event timeline to snapshots.
- **Keyboard Controls (current prototype):** `Space` play/pause animation, `←/→` cycle snapshots, `R` reload directory, `Esc` exit.
