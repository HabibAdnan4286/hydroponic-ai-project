"""Step 9b - Conceptual diagrams (no data needed).

fig_c1_framework        Proposed vision + IoT sensor fusion framework.
                        SOLID boxes = implemented in the case study; DASHED boxes = proposed / future work.
fig_c2_validation_ladder Levels of validation evidence, with where the case study sits.

Edit the text in the script freely - the boxes are only labels. Keep solid vs dashed honest.

Run:  python paper\\scripts\\make_conceptual_figures.py
"""
from matplotlib.patches import FancyBboxPatch

from common import PALETTE, plt, savefig


def box(ax, x, y, w, h, text, proposed=False, fc="#EAF2F8", fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#333333",
                                lw=1.1, ls="--" if proposed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, proposed=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.1, ls="--" if proposed else "-", color="#333333"))


# ---------------- Framework ----------------
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.set_xlim(0, 20.6)
ax.set_ylim(0, 10.4)
ax.axis("off")

ax.text(0.2, 9.6, "Vision branch", fontsize=9, fontweight="bold", color=PALETTE[0])
box(ax, 0.2, 7.6, 3.2, 1.6, "Camera images\n(public leaf dataset\nused in case study)")
box(ax, 4.2, 7.6, 3.2, 1.6, "Pre-processing\n+ augmentation")
box(ax, 8.2, 7.6, 3.2, 1.6, "CNN backbone\n(MobileNetV2, transfer\nlearning)")
arrow(ax, 3.4, 8.4, 4.2, 8.4)
arrow(ax, 7.4, 8.4, 8.2, 8.4)

ax.text(0.2, 5.2, "Sensor branch (IoT node)", fontsize=9, fontweight="bold", color=PALETTE[2])
box(ax, 0.2, 3.2, 3.2, 1.6, "Sensors: temp, humidity,\nTDS/EC, pH (public\ndata used in case study)")
box(ax, 4.2, 3.2, 3.2, 1.6, "Windowing /\nfeature extraction", proposed=True)
box(ax, 8.2, 3.2, 3.2, 1.6, "Tabular / temporal model\n(Random Forest in\ncase study)")
arrow(ax, 3.4, 4.0, 4.2, 4.0, proposed=True)
arrow(ax, 7.4, 4.0, 8.2, 4.0, proposed=True)

box(ax, 12.4, 5.2, 3.0, 2.0, "Fusion module\n(early / late / hybrid)", proposed=True, fc="#FEF5E7", bold=True)
arrow(ax, 11.4, 8.4, 12.4, 6.7, proposed=True)
arrow(ax, 11.4, 4.0, 12.4, 5.7, proposed=True)

box(ax, 16.4, 7.6, 3.4, 1.6, "Stress class\nN / K deficiency, disease\n(+ heat, salinity*)", proposed=False, fc="#E8F6EF")
box(ax, 16.4, 5.2, 3.4, 1.6, "Growth stage /\ndays-to-harvest\n+ uncertainty", proposed=False, fc="#E8F6EF")
box(ax, 16.4, 2.4, 3.4, 1.8, "Decision support\n(alerts, nutrient dosing,\nclimate control)", proposed=True, fc="#FEF5E7")
arrow(ax, 15.4, 6.7, 16.4, 8.4, proposed=True)
arrow(ax, 15.4, 6.2, 16.4, 6.0, proposed=True)
arrow(ax, 18.1, 5.2, 18.1, 4.2, proposed=True)
arrow(ax, 11.4, 8.4, 16.4, 8.9, proposed=False)   # vision-only path (implemented)
arrow(ax, 11.4, 4.0, 16.4, 5.6, proposed=False)   # sensor-only path (implemented)

ax.add_patch(FancyBboxPatch((11.9, 1.2), 8.5, 8.6, boxstyle="round,pad=0.02,rounding_size=0.2", fc="none", ec="#888888", lw=1, ls=":"))
ax.text(12.0, 0.7, "Edge device (e.g. Raspberry Pi / Jetson) - proposed deployment target", fontsize=7.5, color="#555555")
ax.plot([0.5, 1.5], [0.35, 0.35], color="#333333", lw=1.1)
ax.text(1.7, 0.35, "implemented in case study", fontsize=7.5, va="center")
ax.plot([6.0, 7.0], [0.35, 0.35], color="#333333", lw=1.1, ls="--")
ax.text(7.2, 0.35, "proposed / future work", fontsize=7.5, va="center")
ax.text(0.2, 10.1, "* heat and salinity classes are proposed; no labelled data for them was used in the case study.", fontsize=7, color="#555555")
savefig(fig, "fig_c1_framework")

# ---------------- Validation ladder ----------------
levels = [
    ("Level 4  External validation", "new farm, camera, cultivar or season", "tests generalisation", "#009E73"),
    ("Level 3  Temporal hold-out", "train on earlier cycles, test on later ones", "tests drift over time", "#56B4E9"),
    ("Level 2  Group-aware split", "split by plant / unit (no shared plants)", "removes same-plant leakage", "#E69F00"),
    ("Level 1  Random image split", "images shuffled, near-duplicates may straddle the split", "prone to optimistic estimates", "#D55E00"),
]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8.6)
ax.axis("off")
for i, (title, desc, risk, col) in enumerate(levels):
    y = 0.4 + (3 - i) * 2.0
    ax.add_patch(FancyBboxPatch((0.2, y), 6.2, 1.5, boxstyle="round,pad=0.02,rounding_size=0.1", fc=col, ec="#333333", alpha=0.85))
    ax.text(0.45, y + 1.05, title, fontsize=9, fontweight="bold", va="center")
    ax.text(0.45, y + 0.45, desc, fontsize=7.5, va="center")
    ax.text(6.7, y + 0.75, risk, fontsize=7.5, va="center", style="italic")
ax.annotate("", xy=(9.8, 8.0), xytext=(9.8, 0.6), arrowprops=dict(arrowstyle="->", lw=1.3))
ax.text(9.9, 4.3, "Strength of evidence", rotation=90, va="center", ha="left", fontsize=8)
ax.text(0.2, 8.35, "Case study here: vision = Level 2 only to the extent near-duplicates are grouped; sensors = Level 2 (plant-grouped CV).",
        fontsize=7, color="#555555")
savefig(fig, "fig_c2_validation_ladder")
print("Done.")
