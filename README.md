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
  the contribution of wind-aware adaptation, tested under both mild
  and substantially harsher wind conditions.

## Repository structure

```
.
├── uav_energy_sim.py              # Core simulation: dynamics, wind, energy model, guidance law, QP controller
├── run_experiment.py               # Monte Carlo experiment driver and statistics
├── run_harsh_wind.py                # Harsher-wind follow-up experiment (tests whether wind-awareness benefit grows with wind severity)
├── make_figures.py                  # Figure generation from experiment output
├── make_architecture_diagram.py     # Generates the system architecture diagram
├── AEMCF_UAV_simulation.ipynb        # End-to-end Colab notebook (simulation -> experiment -> figures)
└── results/
    ├── results.csv                  # Per-seed, per-controller output, primary experiment (mild wind, 50 seeds)
    ├── harsh_results.csv             # Per-seed, per-controller output, harsher-wind follow-up (30 seeds)
    ├── summary.csv                   # Aggregated summary statistics (primary experiment)
    ├── stats.json                     # Paired significance tests (primary experiment)
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

## Harsher-wind follow-up experiment

A supplementary experiment tests whether the wind-awareness ablation's
endurance contribution grows under stronger wind (mean speed 7 m/s,
max gust 15 m/s, shorter correlation time of 20 s, vs. 4 m/s / 7.5 m/s
/ 55 s in the primary experiment):

```bash
# Run 30 seeds under harsher wind, seeds 0-4 through 25-29 in batches
python3 run_harsh_wind.py 30 0 harsh_results.csv
```

`run_harsh_wind.py` takes three positional arguments: number of seeds,
starting seed offset, and output CSV path, plus an optional `--append`
flag for running in batches. The wind parameters are set as
`HARSH_WIND` at the top of the script and can be edited directly.
Result: under the wind conditions tested, this contribution did *not*
grow with wind severity — see the paper's Results and Discussion for
the full analysis and a candidate explanation.

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
