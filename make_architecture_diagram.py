import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

plt.rcParams["font.family"] = "serif"

fig, ax = plt.subplots(figsize=(9.4, 6.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 66)
ax.axis("off")
ax.set_aspect("equal")

EDGE = "#000000"
TXT = "#000000"
FILL = "#ffffff"
LW = 1.4

def box(x, y, w, h, label_text, fontsize=9.2, weight="normal", fill=FILL, lw=LW):
    b = Rectangle((x, y), w, h, linewidth=lw, edgecolor=EDGE,
                  facecolor=fill, zorder=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, label_text, ha="center", va="center",
            fontsize=fontsize, color=TXT, weight=weight, zorder=3,
            linespacing=1.5)
    return dict(x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y + h / 2)

def straight_arrow(p_from, p_to, lw=1.15):
    a = FancyArrowPatch(p_from, p_to, arrowstyle="-|>", mutation_scale=11,
                         linewidth=lw, color=EDGE, zorder=1,
                         shrinkA=0, shrinkB=0)
    ax.add_patch(a)

def elbow_arrow(points, lw=1.15, dashed=False):
    ls = (0, (5, 3)) if dashed else "-"
    for i in range(len(points) - 2):
        ax.plot([points[i][0], points[i + 1][0]],
                 [points[i][1], points[i + 1][1]],
                 color=EDGE, linewidth=lw, zorder=1, solid_capstyle="butt",
                 linestyle=ls)
    a = FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                         mutation_scale=11, linewidth=lw, color=EDGE,
                         zorder=1, shrinkA=0, shrinkB=0, linestyle=ls)
    ax.add_patch(a)

def dot(x, y, r=0.55):
    ax.add_patch(Circle((x, y), r, facecolor=EDGE, edgecolor=EDGE, zorder=5))

def label(x, y, text, fontsize=7.6, style="italic", ha="center", va="center", bg=True):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=TXT,
            style=style, zorder=4,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2) if bg else None)

# boxes: mission -> guidance -> plant (top), QP (middle), estimator (bottom)
mission = box(2, 48, 17, 12,
              "Mission Waypoints /\nOn-Station Requirement\n(offline input)",
              fontsize=8.6)

guidance = box(25, 48, 24, 14,
               "Wind-Compensating\nGuidance Law\n(closed-form, 1 Hz)",
               fontsize=9.4, weight="bold")

plant = box(63, 48, 32, 14,
            "UAV Point-Mass Dynamics\n+ Hybrid Energy Plant",
            fontsize=9.4, weight="bold")

qp = box(25, 27, 24, 14,
         "Power-Allocation MPC\n(convex QP, OSQP, 1 Hz)",
         fontsize=9.4, weight="bold")

sense = box(25, 4, 70, 14,
            "State & Wind Estimator\n(GPS, IMU, airspeed, magnetometer)\n"
            "outputs: SOC, fuel remaining, position, wind estimate $\\hat{\\mathbf{w}}$",
            fontsize=8.9, weight="bold")

# forward path
straight_arrow((mission["x"] + mission["w"], mission["cy"]),
               (guidance["x"], guidance["cy"]))
label((mission["x"] + mission["w"] + guidance["x"]) / 2, mission["cy"] + 2.6,
      "next\nwaypoint", fontsize=7.2)

straight_arrow((guidance["x"] + guidance["w"], guidance["cy"] + 2.2),
               (plant["x"], plant["cy"] + 2.2))
label((guidance["x"] + guidance["w"] + plant["x"]) / 2, guidance["cy"] + 4.8,
      "$v_{cmd}$, heading,\nclimb rate", fontsize=7.2)

straight_arrow((guidance["cx"] - 5, guidance["y"]),
               (qp["cx"] - 5, qp["y"] + qp["h"]))
label(guidance["cx"] + 13, (guidance["y"] + qp["y"] + qp["h"]) / 2 + 1,
      "power demand\nforecast $P_{total}(k{+}i)$", fontsize=7.2)

qp_out_x = qp["x"] + qp["w"]
route_x = qp_out_x + 6
elbow_arrow([
    (qp_out_x, qp["cy"]),
    (route_x, qp["cy"]),
    (route_x, plant["y"]),
    (plant["x"] + 6, plant["y"]),
])
label(route_x + 4.2, (qp["cy"] + plant["y"]) / 2, "$P_{batt}$, $P_{fc}$",
      fontsize=7.4, ha="left")

# feedback path - each line starts with a dot on the estimator box so
# the source is unambiguous

plant_feed_x = plant["x"] + plant["w"] - 6
dot(plant_feed_x, plant["y"])
elbow_arrow([
    (plant_feed_x, plant["y"] - 0.01),
    (plant_feed_x, sense["y"] + sense["h"]),
])
label(plant_feed_x + 5.5, (plant["y"] + sense["y"] + sense["h"]) / 2,
      "true\nstate", fontsize=7.2)

qp_feed_x = qp["x"] + qp["w"] - 6
dot(qp_feed_x, sense["y"] + sense["h"])
elbow_arrow([
    (qp_feed_x, sense["y"] + sense["h"] + 0.01),
    (qp_feed_x, qp["y"]),
])
label(qp_feed_x + 9.0, (sense["y"] + sense["h"] + qp["y"]) / 2,
      "SOC, fuel\nremaining", fontsize=7.2)

wind_feed_x = sense["x"] + 4
dot(wind_feed_x, sense["y"] + sense["h"])
corridor_x = 13
elbow_arrow([
    (wind_feed_x, sense["y"] + sense["h"] + 0.01),
    (wind_feed_x, sense["y"] + sense["h"] + 4.5),
    (corridor_x, sense["y"] + sense["h"] + 4.5),
    (corridor_x, guidance["cy"] - 3.2),
    (guidance["x"], guidance["cy"] - 3.2),
])
label(corridor_x + 8.5, sense["y"] + sense["h"] + 6.6,
      "$\\hat{\\mathbf{w}}$, SOC", fontsize=7.4, ha="left")

ax.text(corridor_x - 3.4, (sense["y"] + sense["h"] + guidance["cy"]) / 2,
        "feedback from\nState & Wind Estimator\n(closes the loop at 1 Hz)",
        rotation=90, ha="center", va="center", fontsize=6.9,
        color="#333333", style="italic")

plt.tight_layout()
fig.savefig("fig0_architecture.png", dpi=300, bbox_inches="tight")
print("Architecture diagram written.")
