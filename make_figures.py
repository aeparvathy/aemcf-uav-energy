import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
with open("traces.json") as f:
    traces = json.load(f)

N_SEEDS = df.method.value_counts().iloc[0]

METHOD_LABEL = {
    "baseline_fixed": "Baseline 1 (Fixed 50/50)",
    "baseline_offline": "Baseline 2 (Static offline split)",
    "aemcf": "AEMCF (proposed)",
    "aemcf_nowind": "AEMCF (wind term removed)",
}
COLORS = {
    "baseline_fixed": "#9e9e9e",
    "baseline_offline": "#4c72b0",
    "aemcf": "#c44e52",
    "aemcf_nowind": "#dd8452",
}
methods_main = ["baseline_fixed", "baseline_offline", "aemcf"]

# Fig 1: endurance bar chart with individual seeds
fig, ax = plt.subplots(figsize=(6.2, 4.2))
means = [df[df.method == m].endurance_min.mean() for m in methods_main]
stds = [df[df.method == m].endurance_min.std() for m in methods_main]
x = np.arange(len(methods_main))
ax.bar(x, means, yerr=stds, capsize=4, color=[COLORS[m] for m in methods_main], width=0.55)
for i, m in enumerate(methods_main):
    ys = df[df.method == m].endurance_min.values
    xs = np.random.default_rng(0).normal(i, 0.04, size=len(ys))
    ax.scatter(xs, ys, color="black", s=10, zorder=5, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels([METHOD_LABEL[m] for m in methods_main], rotation=12, ha="right")
ax.set_ylabel("Endurance (minutes)")
ax.set_title(f"Endurance across {N_SEEDS} Monte Carlo wind realisations")
fig.tight_layout()
fig.savefig("fig1_endurance.png", dpi=200)
plt.close(fig)

# Fig 2: SOC and fuel traces for one representative run
fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.6), sharex=True)
for m in methods_main:
    soc = np.array(traces[m]["soc"]) * 100.0
    tmin = np.arange(len(soc)) / 60.0
    axes[0].plot(tmin, soc, label=METHOD_LABEL[m], color=COLORS[m], linewidth=1.4)
axes[0].axhline(20, color="k", linestyle=":", linewidth=1, label="SOC floor (20%)")
axes[0].set_ylabel("Battery SOC (%)")
axes[0].legend(fontsize=8, loc="lower left")
axes[0].set_title("Representative mission (seed 0): battery SOC and fuel remaining")

for m in methods_main:
    fuel = np.array(traces[m]["fuel"])
    tmin = np.arange(len(fuel)) / 60.0
    axes[1].plot(tmin, fuel, color=COLORS[m], linewidth=1.4)
axes[1].axhline(0, color="k", linestyle=":", linewidth=1)
axes[1].set_ylabel("Fuel remaining (Wh)")
axes[1].set_xlabel("Time (minutes)")
fig.tight_layout()
fig.savefig("fig2_soc_fuel_traces.png", dpi=200)
plt.close(fig)

# Fig 3: energy vs mean wind speed
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for m in methods_main:
    sub = df[df.method == m]
    ax.scatter(sub.mean_wind, sub.energy_Wh, color=COLORS[m], label=METHOD_LABEL[m], s=28, alpha=0.8)
    if len(sub) > 2:
        z = np.polyfit(sub.mean_wind, sub.energy_Wh, 1)
        xs = np.linspace(sub.mean_wind.min(), sub.mean_wind.max(), 20)
        ax.plot(xs, np.polyval(z, xs), color=COLORS[m], linestyle="--", linewidth=1)
ax.set_xlabel("Mean wind speed over mission (m/s)")
ax.set_ylabel("Total energy consumed (Wh)")
ax.set_title(f"Energy consumption vs. mean wind speed ({N_SEEDS} seeds)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig3_energy_vs_wind.png", dpi=200)
plt.close(fig)

# Fig 4: on-station loiter vs a target requirement
df["loiter_min"] = df["endurance_min"] - df["mission_time_min"]
REQ = 80.0
fig, ax = plt.subplots(figsize=(6.2, 4.2))
means_l = [df[df.method == m].loiter_min.mean() for m in methods_main]
stds_l = [df[df.method == m].loiter_min.std() for m in methods_main]
ax.bar(x, means_l, yerr=stds_l, capsize=4, color=[COLORS[m] for m in methods_main], width=0.55)
ax.axhline(REQ, color="k", linestyle=":", linewidth=1.3, label=f"On-station requirement ({REQ:.0f} min)")
ax.set_xticks(x)
ax.set_xticklabels([METHOD_LABEL[m] for m in methods_main], rotation=12, ha="right")
ax.set_ylabel("On-station loiter duration (minutes)")
ax.set_title("On-station endurance after completing the search pattern")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig4_loiter_vs_requirement.png", dpi=200)
plt.close(fig)

print("Figures written.")
for m in methods_main:
    sub = df[df.method == m]
    meets_req = (sub.loiter_min >= REQ).mean()
    print(f"{m}: loiter mean={sub.loiter_min.mean():.1f} std={sub.loiter_min.std():.2f} meets80={meets_req:.2f}")
