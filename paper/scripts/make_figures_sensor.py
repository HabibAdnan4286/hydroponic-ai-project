"""Step 8b - All sensor-model figures (run sensor_analysis.py first).

Makes:
  fig_s1_pred_vs_actual    out-of-fold predicted vs actual growth day (plants never seen in training)
  fig_s2_residuals         residuals by growth day (shows where the model fails)
  fig_s3_perm_importance   permutation importance on held-out plants (mean +/- SD across folds)
  fig_s4_correlation       correlation heatmap of the four sensor features, calendar date and growth day
  fig_s5_timeseries        raw sensor traces for a few plants
  fig_s6_confound          (a) plant timelines, (b) temperature vs growth day by start-month cohort,
                           (c) R2 by feature set incl. calendar-date-only model  <- the key figure for the limitation section

Run:  python paper\\scripts\\make_figures_sensor.py
"""
import numpy as np
import pandas as pd

from common import PALETTE, TAB, load_sensor, plt, savefig

need = ["sensor_oof_all4_rf.csv", "sensor_perm_importance.csv", "sensor_cv_summary.csv",
        "sensor_plant_start_dates.csv", "sensor_cohort_temp.csv"]
missing = [n for n in need if not (TAB / n).exists()]
if missing:
    raise SystemExit(f"Missing {missing} - run sensor_analysis.py first.")

df = load_sensor()
oof = pd.read_csv(TAB / "sensor_oof_all4_rf.csv")
perm = pd.read_csv(TAB / "sensor_perm_importance.csv")
summ = pd.read_csv(TAB / "sensor_cv_summary.csv")
plants = pd.read_csv(TAB / "sensor_plant_start_dates.csv", parse_dates=["start_date", "end_date"])
cohort = pd.read_csv(TAB / "sensor_cohort_temp.csv")
NICE = {"temp_c": "Temperature (C)", "humidity": "Humidity (%)", "tds": "TDS (ppm)", "ph": "pH",
        "growth_day": "Growth day", "date_index": "Calendar date index"}

# ---- S1 predicted vs actual ----
fig, ax = plt.subplots(figsize=(3.8, 3.6))
ax.scatter(oof.growth_day, oof.pred, s=5, alpha=0.3, color=PALETTE[0])
lim = [0, max(oof.growth_day.max(), oof.pred.max()) + 2]
ax.plot(lim, lim, "k--", lw=0.8, label="perfect prediction")
ss_res, ss_tot = ((oof.growth_day - oof.pred) ** 2).sum(), ((oof.growth_day - oof.growth_day.mean()) ** 2).sum()
ax.text(0.04, 0.93, f"R$^2$ = {1 - ss_res / ss_tot:.2f}\nMAE = {np.abs(oof.growth_day - oof.pred).mean():.1f} days",
        transform=ax.transAxes, va="top", fontsize=8)
ax.set_xlabel("Actual growth day")
ax.set_ylabel("Predicted growth day (held-out plants)")
ax.legend(frameon=False, loc="lower right")
savefig(fig, "fig_s1_pred_vs_actual")

# ---- S2 residuals ----
oof["resid"] = oof.pred - oof.growth_day
b = oof.groupby(pd.cut(oof.growth_day, bins=np.arange(0, 54, 6))).resid.agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(4.4, 3))
ax.scatter(oof.growth_day, oof.resid, s=4, alpha=0.2, color=PALETTE[0])
mids = [i.mid for i in b.index]
ax.errorbar(mids, b["mean"], yerr=b["std"], color=PALETTE[1], marker="o", ms=3, capsize=2, lw=1, label="bin mean +/- SD")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Actual growth day")
ax.set_ylabel("Residual (predicted - actual, days)")
ax.legend(frameon=False)
savefig(fig, "fig_s2_residuals")

# ---- S3 permutation importance ----
pi = perm.groupby("feature").importance.agg(["mean", "std"]).reindex(["temp_c", "humidity", "tds", "ph"])
fig, ax = plt.subplots(figsize=(3.8, 2.6))
ax.barh([NICE[f] for f in pi.index][::-1], pi["mean"][::-1], xerr=pi["std"][::-1], color=PALETTE[0], capsize=2)
ax.set_xlabel("Drop in R$^2$ when feature is shuffled (held-out plants)")
savefig(fig, "fig_s3_perm_importance")

# ---- S4 correlation heatmap ----
cols = ["temp_c", "humidity", "tds", "ph", "date_index", "growth_day"]
cm = df[cols].corr()
fig, ax = plt.subplots(figsize=(4.4, 3.8))
im = ax.imshow(cm, vmin=-1, vmax=1, cmap="RdBu_r")
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels([NICE[c] for c in cols], rotation=40, ha="right")
ax.set_yticklabels([NICE[c] for c in cols])
ax.spines[:].set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
savefig(fig, "fig_s4_correlation")

# ---- S5 time series ----
pick = np.linspace(df.plant_id.min(), df.plant_id.max(), 5).round().astype(int)
pick = [p for p in pick if p in set(df.plant_id)]
fig, axes = plt.subplots(2, 2, figsize=(7, 4.4), sharex=True)
for ax, c in zip(axes.ravel(), ["temp_c", "humidity", "tds", "ph"]):
    for k, p in enumerate(pick):
        g = df[df.plant_id == p]
        ax.plot(g.growth_day, g[c], lw=0.9, color=PALETTE[k % len(PALETTE)], label=f"plant {p}")
    ax.set_ylabel(NICE[c])
for ax in axes[1]:
    ax.set_xlabel("Growth day")
axes[0, 0].legend(frameon=False, ncol=2, fontsize=7)
savefig(fig, "fig_s5_timeseries")

# ---- S6 confound panel ----
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
ax = axes[0]
pp = plants.sort_values("start_date").reset_index(drop=True)
ax.hlines(range(len(pp)), pp.start_date, pp.end_date, color=PALETTE[0], lw=1.2)
ax.set_xlabel("Calendar date")
ax.set_ylabel("Plant (sorted by start date)")
ax.set_title("(a) When each plant was grown")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax = axes[1]
for k, (m, g) in enumerate(cohort.groupby("start_month")):
    ax.plot(g.growth_day, g.temp_c, color=PALETTE[k % len(PALETTE)], lw=1.2, label=m)
ax.set_xlabel("Growth day")
ax.set_ylabel("Mean temperature (C)")
ax.set_title("(b) Temperature vs growth day,\nby start-month cohort")
ax.legend(frameon=False, fontsize=6, ncol=2, title="start month", title_fontsize=7)
ax = axes[2]
s = summ[((summ.model == "RF") | (summ.model == "Dummy"))].copy()
order = ["mean baseline", "date_only (calendar)", "temp_only", "no_temp (hum+tds+pH)", "all4 (temp+hum+tds+pH)", "all4 + calendar date"]
s = s[s.features.isin(order)].set_index("features").reindex(order)
cols_bar = [PALETTE[7], PALETTE[1], PALETTE[0], PALETTE[0], PALETTE[0], PALETTE[1]]
ax.barh(range(len(s))[::-1], s.r2_mean, xerr=s.r2_sd, color=cols_bar, capsize=2)
ax.set_yticks(range(len(s))[::-1])
ax.set_yticklabels(order, fontsize=7)
ax.set_xlabel("R$^2$ on held-out plants (mean +/- SD)")
ax.set_title("(c) Can calendar date alone\npredict growth day?")
savefig(fig, "fig_s6_confound")

print("\nDone. Next: python paper\\scripts\\make_tables.py")
