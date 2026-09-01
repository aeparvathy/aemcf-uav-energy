"""
Monte Carlo driver for the AEMCF simulation.

Usage:
    python3 run_experiment.py --n_runs 50 --out results.csv

Runs baseline_fixed, baseline_offline, aemcf, aemcf_nowind across
n_runs wind seeds (same seeds for every method, so paired tests are
valid). Saves per-run results to CSV plus a SOC/fuel trace for seed 0.
"""
import argparse
import time
import json
import numpy as np
import pandas as pd
from scipy import stats

import uav_energy_sim as sim

METHODS = ["baseline_fixed", "baseline_offline", "aemcf", "aemcf_nowind"]


def run_all(n_runs, max_time=13000.0, seed0=0, verbose=True, out_csv=None, append=False, wind_kwargs=None):
    mpc = sim.PowerMPC(w_balance=15.0, w_fc_band=0.0, w_du=0.02)
    rows = []
    traces = {}
    t_start = time.time()
    header_written = append and out_csv is not None and __import__("os").path.exists(out_csv)
    for i in range(n_runs):
        seed = seed0 + i
        seed_rows = []
        for m in METHODS:
            r = sim.run_mission(m, seed=seed, mpc=mpc, max_time=max_time, wind_kwargs=wind_kwargs)
            if seed == seed0 and not append:
                traces[m] = {"soc": r["soc_trace"], "fuel": r["fuel_trace"]}
            r = {k: v for k, v in r.items() if k not in ("soc_trace", "fuel_trace")}
            seed_rows.append(r)
        rows.extend(seed_rows)
        if out_csv is not None:
            pd.DataFrame(seed_rows).to_csv(out_csv, mode="a" if (append or i > 0) else "w",
                                            header=not header_written, index=False)
            header_written = True
        if verbose:
            elapsed = time.time() - t_start
            print(f"  seed {seed} done ({i+1}/{n_runs}), elapsed {elapsed:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    return df, traces


def summarize(df):
    summary = df.groupby("method").agg(
        endurance_mean=("endurance_min", "mean"),
        endurance_std=("endurance_min", "std"),
        energy_mean=("energy_Wh", "mean"),
        energy_std=("energy_Wh", "std"),
        completion_rate=("completed", "mean"),
        comp_time_mean=("mean_comp_time", "mean"),
    )
    summary["completion_rate"] *= 100.0
    return summary


def paired_tests(df):
    """Paired t-test: AEMCF vs the best baseline, matched by seed."""
    piv_end = df.pivot(index="seed", columns="method", values="endurance_min")
    piv_energy = df.pivot(index="seed", columns="method", values="energy_Wh")
    baselines = ["baseline_fixed", "baseline_offline"]
    best_baseline = piv_end[baselines].mean().idxmax()

    t_end, p_end = stats.ttest_rel(piv_end["aemcf"], piv_end[best_baseline])
    t_en, p_en = stats.ttest_rel(piv_energy["aemcf"], piv_energy[best_baseline])
    t_abl, p_abl = stats.ttest_rel(piv_end["aemcf"], piv_end["aemcf_nowind"])
    return {
        "best_baseline": best_baseline,
        "endurance_vs_best_baseline": {"t": float(t_end), "p": float(p_end)},
        "energy_vs_best_baseline": {"t": float(t_en), "p": float(p_en)},
        "ablation_wind_term_endurance": {"t": float(t_abl), "p": float(p_abl)},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_runs", type=int, default=15)
    ap.add_argument("--max_time", type=float, default=13000.0)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", type=str, default="results.csv")
    ap.add_argument("--traces_out", type=str, default="traces.json")
    ap.add_argument("--summary_out", type=str, default="summary.csv")
    ap.add_argument("--stats_out", type=str, default="stats.json")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    df, traces = run_all(args.n_runs, max_time=args.max_time, seed0=args.seed0,
                          out_csv=args.out, append=args.append)
    summary = summarize(df)
    summary.to_csv(args.summary_out)
    tests = paired_tests(df)
    with open(args.stats_out, "w") as f:
        json.dump(tests, f, indent=2)
    with open(args.traces_out, "w") as f:
        json.dump(traces, f)

    print("\n=== SUMMARY ===")
    print(summary)
    print("\n=== SIGNIFICANCE TESTS (paired, n=%d) ===" % args.n_runs)
    print(json.dumps(tests, indent=2))
