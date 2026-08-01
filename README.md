# AEMCF: Adaptive Energy Management and Control Framework for Long-Endurance UAVs

Simulation code for a hybrid battery/fuel-cell UAV energy management
system. The controller combines a wind-compensating guidance law with
a receding-horizon, convex quadratic-program (QP) power-allocation
model predictive controller (MPC), coordinated through a shared state
estimate (battery state of charge, remaining fuel budget, position,
wind).

## Overview

- **Vehicle model**: 3-DOF point-mass fixed-wing UAV dynamics.
- **Wind model**: first-order Gauss–Markov (Ornstein–Uhlenbeck)
  horizontal gust process.
- **Energy model**: battery state-of-charge dynamics with discharge
  efficiency, and a fuel cell with a slew-rate limit, a finite onboard
  fuel budget, and a load-dependent electrical efficiency curve.
- **Guidance law**: closed-form, wind-compensating crab-angle
  correction with SOC-aware airspeed scheduling.
- **Power-allocation MPC**: a genuine convex QP (no linearisation
  error), solved every control step with
  [OSQP](https://osqp.org/) via [CVXPY](https://www.cvxpy.org/), using
  an equivalent-consumption objective that balances battery and fuel
  depletion against each other.
- **Baselines**: a fixed 50/50 battery/fuel-cell power split, and a
  static equivalent-consumption split computed once before flight.
- **Ablation**: a wind-blind variant of the controller, for isolating
  the contribution of wind-aware adaptation.

## Repository structure

```
.
├── uav_energy_sim.py              # Core simulation: dynamics, wind, energy model, guidance law, QP controller
├── run_experiment.py               # Monte Carlo experiment driver and statistics
├── make_figures.py                  # Figure generation from experiment output
├── make_architecture_diagram.py     # Generates the system architecture diagram
├── AEMCF_UAV_simulation.ipynb        # End-to-end Colab notebook (simulation -> experiment -> figures)
└── results/
    ├── results.csv                  # Per-seed, per-controller experiment output
    ├── summary.csv                   # Aggregated summary statistics
    ├── stats.json                     # Paired significance tests
    └── traces.json                    # SOC / fuel time series for a representative run
```

## Quick start (Colab)

Open `AEMCF_UAV_simulation.ipynb` in Google Colab and run all cells.
The notebook installs its own dependencies, writes out the simulation
modules, runs a Monte Carlo experiment, and produces all figures
inline.

## Quick start (local)

```bash
pip install cvxpy numpy pandas scipy matplotlib

# Run a Monte Carlo experiment (50 wind seeds, all four controllers)
python3 run_experiment.py --n_runs 50 --out results.csv \
    --summary_out summary.csv --stats_out stats.json --traces_out traces.json

# Generate figures from the results
python3 make_figures.py

# Generate the architecture diagram
python3 make_architecture_diagram.py
```

`run_experiment.py --help` lists all options (seed offset, mission
duration cap, output paths).

## Requirements

- Python 3.9+
- `cvxpy` (with the bundled OSQP solver)
- `numpy`, `pandas`, `scipy`, `matplotlib`

## Extending the simulation

- **Wind conditions**: edit the `WindProcess` class in
  `uav_energy_sim.py` (`mean_speed`, `max_speed`, `tau`).
- **Vehicle / battery / fuel-cell parameters**: module-level constants
  near the top of `uav_energy_sim.py`.
- **Mission geometry**: `default_mission()` defines the waypoint list;
  `LOITER_RADIUS` sets the post-mission holding-pattern radius.
- **QP weights**: `PowerMPC(w_balance=..., w_fc_band=..., w_du=...)`,
  instantiated in `run_experiment.run_all`.

## License

Add a license of your choice (e.g. MIT) before making the repository
public.
