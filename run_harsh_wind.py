import sys
import pandas as pd
import uav_energy_sim as sim
import run_experiment as rexp

HARSH_WIND = dict(mean_speed=7.0, max_speed=15.0, tau=20.0, sigma=2.5)

if __name__ == "__main__":
    n_runs = int(sys.argv[1])
    seed0 = int(sys.argv[2])
    out_csv = sys.argv[3]
    append = "--append" in sys.argv

    df, traces = rexp.run_all(n_runs, max_time=13000.0, seed0=seed0, verbose=True,
                               out_csv=out_csv, append=append, wind_kwargs=HARSH_WIND)
    print(df.groupby("method")[["endurance_min", "energy_Wh", "mean_wind", "max_wind"]].mean())
