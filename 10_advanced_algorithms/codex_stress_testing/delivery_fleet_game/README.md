# Delivery Fleet Management System

Interactive simulation for managing a delivery company's fleet, balancing capacity planning, routing optimization, and financial health. Built as part of the *Art of Programming – Advanced Algorithms* series.

## Project Goals
- Model vehicles, packages, routes, and the overall company state through clean Python architecture.
- Explore classic and advanced routing algorithms by treating them as pluggable "agents".
- Provide both educational insight and an engaging management experience via CLI in Phase 1, Pygame visuals in later phases.

## Phase Roadmap
1. **Foundation (current):** Core models, JSON loaders, day cycle, manual planning, financial ledger, baseline tests.
2. **Agents & Optimization:** Greedy, Backtracking, and additional strategies with comparative analytics.
3. **Visualization & Polish:** Pygame interface, interactive dashboards, persistence enhancements.

## Getting Started
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m delivery_fleet_game.main
```

- The game auto-loads the most recent file from `savegames/` if present; otherwise it falls back to the seed template.
- Use the CLI to assign packages manually or run the `Greedy Agent` for a first-fit plan, then validate/simulate to finish the day.
- After completing a day or exiting with autosave, snapshots land in `snapshots/`. Launch the work-in-progress viewer:
  ```bash
  python -m delivery_fleet_game.preview
  ```
  Controls: `←/→` cycle between days, `R` reload snapshot files, `Esc` quits.

## Repository Layout
```
delivery_fleet_game/
├── data/                 # Seed JSON data for map, vehicles, packages
├── savegames/            # Generated save files (ignored in git)
├── src/                  # Application source modules
│   ├── ui/               # Visualization prototype (Pygame)
├── tests/                # pytest test suite + fixtures
└── docs/                 # Design and planning artifacts
```
