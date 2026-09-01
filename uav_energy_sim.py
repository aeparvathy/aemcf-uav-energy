"""
AEMCF simulation: hybrid battery/fuel-cell UAV energy management.

Two layers:
1. Guidance law - nonlinear, closed-form, wind-compensating heading + speed
2. Power-allocation QP - convex, solved every step via OSQP/cvxpy

Also includes two baselines (fixed 50/50 split, static offline split)
and a wind-blind ablation of AEMCF.

Units: SI (m, s, kg, W, J), except energy in Wh and SOC as a fraction.
"""

import numpy as np
import cvxpy as cp
import time

# --- vehicle / battery / fuel cell params (illustrative, not a real airframe) ---
MASS = 5.5                 # kg
WING_AREA = 0.8             # m^2
RHO = 1.225                 # air density, kg/m^3
CD = 0.045                  # drag coefficient
ETA_PROP = 0.75              # propulsion efficiency
G = 9.81

BATT_VNOM = 22.2             # V
BATT_CAP_AH = 6.0             # Ah
BATT_CAP_WH = BATT_VNOM * BATT_CAP_AH   # 133.2 Wh
ETA_DIS = 0.95               # battery discharge efficiency

FC_RATED = 120.0             # W
FC_MIN = 0.0
FC_MAX = FC_RATED
FC_RAMP = 15.0                # W/s slew limit
FC_EFF_TARGET = 0.70          # most efficient load point
FC_FUEL_WH = 250.0            # onboard fuel budget, Wh
FC_ETA_PEAK = 0.55
FC_ETA_CURVATURE = 0.22

def fc_efficiency(p_fc):
    """PEM fuel cell efficiency curve, peaks near FC_EFF_TARGET."""
    load_frac = np.clip(p_fc, 1e-3, FC_RATED) / FC_RATED
    eta = FC_ETA_PEAK - FC_ETA_CURVATURE * (load_frac - FC_EFF_TARGET) ** 2
    return float(np.clip(eta, 0.15, FC_ETA_PEAK))

P_PAYLOAD = 10.0             # W
P_COMM = 5.0                  # W

V_CRUISE = 15.0               # m/s
V_MIN = 10.0
V_MAX = 20.0
GAMMA_MAX = np.deg2rad(8.0)     # max climb/descent angle

SOC_MIN = 0.20
SOC_MAX = 0.95
SOC_INIT = 0.95

DT = 1.0                      # sim step, s
HORIZON_N = 15                 # MPC horizon, steps

WAYPOINT_CAPTURE_RADIUS = 150.0  # m


def default_mission():
    # 8 waypoints, ~20x20 km area, altitude 150-300 m
    pts = np.array([
        [0,     0,    150],
        [3000,  4200, 220],
        [7500,  2500, 180],
        [11000, 6000, 300],
        [14500, 3200, 200],
        [17000, 8000, 260],
        [19000, 12000, 180],
        [20000, 18000, 150],
    ], dtype=float)
    return pts

LOITER_RADIUS = 300.0  # holding pattern radius after last waypoint, m


class WindProcess:
    """Simplified Gauss-Markov (OU) gust model, not a full Dryden spectrum."""

    def __init__(self, rng, mean_speed=4.0, max_speed=7.5, tau=55.0, dt=DT, sigma=1.6):
        self.rng = rng
        self.tau = tau
        self.dt = dt
        self.max_speed = max_speed
        theta0 = rng.uniform(0, 2 * np.pi)
        speed0 = np.clip(rng.normal(mean_speed, 1.5), 0, max_speed)
        self.w = np.array([speed0 * np.cos(theta0), speed0 * np.sin(theta0)])
        self.sigma = sigma

    def step(self):
        a = np.exp(-self.dt / self.tau)
        noise = self.rng.normal(0, self.sigma * np.sqrt(1 - a ** 2), size=2)
        self.w = a * self.w + noise
        speed = np.linalg.norm(self.w)
        if speed > self.max_speed:
            self.w = self.w / speed * self.max_speed
        return self.w.copy()


def drag_force(v_air):
    return 0.5 * RHO * v_air ** 2 * CD * WING_AREA

def propulsion_power(v_air, climb_rate):
    D = drag_force(v_air)
    p_level = D * v_air / ETA_PROP
    p_climb = max(0.0, MASS * G * climb_rate) / ETA_PROP  # no regen on descent
    return p_level + p_climb

def total_power(v_air, climb_rate):
    return propulsion_power(v_air, climb_rate) + P_PAYLOAD + P_COMM


def guidance_step(pos, wp, wind_est, soc, wind_aware=True, speed_schedule=True):
    """Nonlinear guidance law, solved in closed form (not part of the QP)."""
    to_wp = wp - pos
    dist_xy = np.linalg.norm(to_wp[:2])
    track_dir = to_wp[:2] / (dist_xy + 1e-6)

    w = wind_est if wind_aware else np.zeros(2)

    v_air_cmd = V_CRUISE
    if speed_schedule:
        # slow into headwind when SOC is low, speed up with tailwind when SOC is healthy
        headwind_component = -np.dot(w, track_dir)
        soc_factor = np.clip((soc - SOC_MIN) / (SOC_MAX - SOC_MIN), 0, 1)
        adj = -0.35 * headwind_component / 7.5 * (1.3 - soc_factor)
        v_air_cmd = np.clip(V_CRUISE * (1 + adj * 0.2), V_MIN, V_MAX)

    # crab angle so ground track matches track_dir given wind
    wperp = w[0] * (-track_dir[1]) + w[1] * track_dir[0]
    ratio = np.clip(wperp / max(v_air_cmd, 1e-3), -0.98, 0.98)
    crab = np.arcsin(ratio)
    psi = np.arctan2(track_dir[1], track_dir[0]) + crab

    heading_vec = np.array([np.cos(psi), np.sin(psi)])
    ground_vec_xy = v_air_cmd * heading_vec + w

    dz = to_wp[2]
    horiz_time = max(dist_xy / max(np.linalg.norm(ground_vec_xy), 1e-3), 1.0)
    desired_climb = np.clip(dz / horiz_time, -v_air_cmd * np.sin(GAMMA_MAX),
                             v_air_cmd * np.sin(GAMMA_MAX))

    reached = dist_xy < WAYPOINT_CAPTURE_RADIUS
    return psi, v_air_cmd, ground_vec_xy, desired_climb, reached


def loiter_step(pos, center, wind_est, soc, wind_aware=True, speed_schedule=True):
    """Circular holding pattern around `center`, radius LOITER_RADIUS."""
    w = wind_est if wind_aware else np.zeros(2)
    radial = pos[:2] - center[:2]
    r = np.linalg.norm(radial)
    radial_dir = radial / (r + 1e-6)
    tangent_dir = np.array([-radial_dir[1], radial_dir[0]])

    v_air_cmd = V_CRUISE
    if speed_schedule:
        headwind_component = -np.dot(w, tangent_dir)
        soc_factor = np.clip((soc - SOC_MIN) / (SOC_MAX - SOC_MIN), 0, 1)
        adj = -0.35 * headwind_component / 7.5 * (1.3 - soc_factor)
        v_air_cmd = np.clip(V_CRUISE * (1 + adj * 0.2), V_MIN, V_MAX)

    radial_err = (LOITER_RADIUS - r)
    k_radial = 0.15
    desired_dir = tangent_dir + k_radial * (radial_err / LOITER_RADIUS) * (-radial_dir)
    desired_dir = desired_dir / (np.linalg.norm(desired_dir) + 1e-9)

    wperp = w[0] * (-desired_dir[1]) + w[1] * desired_dir[0]
    ratio = np.clip(wperp / max(v_air_cmd, 1e-3), -0.98, 0.98)
    crab = np.arcsin(ratio)
    psi = np.arctan2(desired_dir[1], desired_dir[0]) + crab
    heading_vec = np.array([np.cos(psi), np.sin(psi)])
    ground_vec_xy = v_air_cmd * heading_vec + w
    return ground_vec_xy, 0.0, v_air_cmd


BATT_USABLE_WH = BATT_CAP_WH * (SOC_MAX - SOC_MIN)
FC_ETA_NOMINAL = 0.50  # nominal efficiency used inside the QP forecast only;
                        # actual fuel accounting uses the real fc_efficiency() curve

class PowerMPC:
    """Convex QP for battery/fuel-cell power split, solved every step (OSQP/cvxpy).

    Balances fractional depletion of battery vs. fuel budget (ECMS-style),
    so one source isn't drained while the other still has capacity.
    """

    def __init__(self, N=HORIZON_N, dt=DT,
                 w_balance=15.0, w_fc_band=0.0, w_du=0.02):
        self.N = N
        self.dt = dt
        self.P_batt = cp.Variable(N)
        self.P_demand = cp.Parameter(N)
        self.soc0 = cp.Parameter()
        self.fuel0 = cp.Parameter()
        self.fc_prev = cp.Parameter()

        k_soc = ETA_DIS * dt / (BATT_CAP_WH * 3600.0)
        k_fuel = dt / (3600.0 * FC_ETA_NOMINAL)

        fc = self.P_demand - self.P_batt
        soc_vec = self.soc0 - k_soc * cp.cumsum(self.P_batt)
        fuel_vec = self.fuel0 - k_fuel * cp.cumsum(fc)
        batt_stress = (SOC_INIT - soc_vec) / (SOC_INIT - SOC_MIN)
        fuel_stress = (FC_FUEL_WH - fuel_vec) / FC_FUEL_WH

        cost = w_balance * cp.sum_squares(batt_stress - fuel_stress)
        if w_fc_band > 0:
            cost += w_fc_band * cp.sum_squares(fc / FC_RATED - FC_EFF_TARGET)
        if w_du > 0 and N > 1:
            cost += w_du * cp.sum_squares(cp.diff(self.P_batt))

        fc_full = cp.hstack([cp.reshape(self.fc_prev, (1,), order='C'), fc])
        constraints = [
            fc >= FC_MIN, fc <= FC_MAX,
            cp.diff(fc_full) <= FC_RAMP * dt,
            cp.diff(fc_full) >= -FC_RAMP * dt,
            self.P_batt >= 0, self.P_batt <= self.P_demand,
        ]

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, soc0, fuel0, fc_prev, p_demand_forecast):
        self.soc0.value = soc0
        self.fuel0.value = fuel0
        self.fc_prev.value = fc_prev
        self.P_demand.value = p_demand_forecast
        t0 = time.perf_counter()
        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                                max_iter=4000)
            p_batt0 = self.P_batt.value[0]
        except Exception:
            p_batt0 = 0.5 * p_demand_forecast[0]
        elapsed = time.perf_counter() - t0
        if p_batt0 is None or np.isnan(p_batt0):
            p_batt0 = 0.5 * p_demand_forecast[0]
        p_batt0 = float(np.clip(p_batt0, 0, p_demand_forecast[0]))
        return p_batt0, elapsed


def run_mission(method, seed, max_time=12000.0, mission=None, mpc=None, wind_kwargs=None):
    """method: 'baseline_fixed', 'baseline_offline', 'aemcf', or 'aemcf_nowind'."""
    rng = np.random.default_rng(seed)
    wind = WindProcess(rng, **(wind_kwargs or {}))
    wp_list = mission if mission is not None else default_mission()

    pos = wp_list[0].copy()
    soc = SOC_INIT
    fc_prev = 0.5 * (propulsion_power(V_CRUISE, 0) + P_PAYLOAD + P_COMM)
    wp_idx = 1
    t = 0.0
    energy_J = 0.0
    fuel_used_Wh = 0.0
    comp_times = []
    soc_trace = []
    fuel_trace = []
    wind_speeds = []

    wind_aware = method in ("aemcf",)
    speed_schedule = method in ("aemcf",)
    use_mpc = method in ("aemcf", "aemcf_nowind")

    # offline baseline: static split computed once from nominal cruise demand
    if method == "baseline_offline":
        p_tot0 = total_power(V_CRUISE, 0)
        denom = ETA_DIS * BATT_USABLE_WH + FC_ETA_NOMINAL * FC_FUEL_WH
        fixed_batt_ratio = (ETA_DIS * BATT_USABLE_WH / denom)
    elif method == "baseline_fixed":
        fixed_batt_ratio = 0.5
    else:
        fixed_batt_ratio = None

    mission_complete_time = None

    while t < max_time and soc > SOC_MIN:
        w_true = wind.step()
        wind_speeds.append(np.linalg.norm(w_true))
        w_est = w_true + rng.normal(0, 0.3, size=2)  # sensor noise
        w_est_used = w_est if (wind_aware) else np.zeros(2)

        in_search_phase = wp_idx < len(wp_list)

        if in_search_phase:
            target = wp_list[wp_idx]
            psi, v_air_cmd, ground_vec_xy, climb_rate, reached = guidance_step(
                pos, target, w_est_used, soc,
                wind_aware=wind_aware, speed_schedule=speed_schedule
            )
            if method in ("baseline_fixed", "baseline_offline"):
                # baselines fly straight, no wind compensation
                to_wp = target - pos
                dist_xy = np.linalg.norm(to_wp[:2])
                track_dir = to_wp[:2] / (dist_xy + 1e-6)
                v_air_cmd = V_CRUISE
                heading_vec = track_dir
                ground_vec_xy = v_air_cmd * heading_vec + w_true
                horiz_time = max(dist_xy / max(np.linalg.norm(ground_vec_xy), 1e-3), 1.0)
                climb_rate = np.clip(to_wp[2] / horiz_time,
                                      -v_air_cmd * np.sin(GAMMA_MAX),
                                      v_air_cmd * np.sin(GAMMA_MAX))
                reached = dist_xy < WAYPOINT_CAPTURE_RADIUS
        else:
            # waypoints done, loiter until battery or fuel runs out
            if mission_complete_time is None:
                mission_complete_time = t
            center = wp_list[-1]
            w_used_loiter = w_true if method in ("baseline_fixed", "baseline_offline") else w_est_used
            ground_vec_xy, climb_rate, v_air_cmd = loiter_step(
                pos, center, w_used_loiter, soc,
                wind_aware=wind_aware if method in ("aemcf", "aemcf_nowind") else False,
                speed_schedule=speed_schedule if method in ("aemcf",) else False,
            )
            reached = False

        p_total = total_power(v_air_cmd, climb_rate)

        if use_mpc:
            forecast = np.full(HORIZON_N, p_total)
            fuel_remaining_now = FC_FUEL_WH - fuel_used_Wh
            p_batt, ct = mpc.solve(soc, fuel_remaining_now, fc_prev, forecast)
            comp_times.append(ct)
        else:
            p_batt = fixed_batt_ratio * p_total
            p_batt = min(p_batt, p_total)

        p_fc = p_total - p_batt
        p_fc = float(np.clip(p_fc, 0, FC_MAX))
        p_batt = p_total - p_fc

        soc -= (p_batt * DT * ETA_DIS) / (BATT_CAP_WH * 3600.0)
        fc_prev = p_fc
        energy_J += p_total * DT

        if p_fc > 1e-6:
            eta_fc = fc_efficiency(p_fc)
            fuel_used_Wh += (p_fc * DT / 3600.0) / eta_fc
        fuel_remaining = FC_FUEL_WH - fuel_used_Wh
        fuel_trace.append(fuel_remaining)

        pos = pos + np.array([ground_vec_xy[0], ground_vec_xy[1], climb_rate]) * DT
        t += DT
        soc_trace.append(soc)

        if in_search_phase and reached:
            wp_idx += 1
        if soc <= SOC_MIN or fuel_remaining <= 0:
            break

    completed = (wp_idx >= len(wp_list))
    energy_Wh = energy_J / 3600.0
    result = {
        "method": method,
        "seed": seed,
        "endurance_min": t / 60.0,
        "mission_time_min": (mission_complete_time / 60.0) if mission_complete_time else np.nan,
        "energy_Wh": energy_Wh,
        "completed": bool(completed),
        "mean_comp_time": float(np.mean(comp_times)) if comp_times else np.nan,
        "final_soc": soc,
        "fuel_used_Wh": fuel_used_Wh,
        "fuel_remaining_Wh": FC_FUEL_WH - fuel_used_Wh,
        "mean_wind": float(np.mean(wind_speeds)) if wind_speeds else np.nan,
        "max_wind": float(np.max(wind_speeds)) if wind_speeds else np.nan,
        "soc_trace": soc_trace,
        "fuel_trace": fuel_trace,
    }
    return result
