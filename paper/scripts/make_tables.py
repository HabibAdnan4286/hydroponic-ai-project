"""Step 9a - Paper tables for the case study (CSV + LaTeX, ready to paste).

Table 1  Dataset summary (both datasets)
Table 2  Vision cross-validation results (mean +/- SD, bootstrap CI, McNemar test)
Table 3  Per-class recall (mean +/- SD across folds)
Table 4  Sensor model: feature-set / model comparison under plant-grouped CV

Run after cv_vision.py and sensor_analysis.py:
    python paper\\scripts\\make_tables.py
Any table whose inputs are missing is skipped with a message.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

from common import TAB, list_images, load_sensor, mean_sd, write_table

# ---------------- Table 1: datasets ----------------
imgs = list_images()
counts = imgs.label.value_counts().sort_index()
sens = load_sensor()
t1 = pd.DataFrame([
    ["Leaf images (Rathor, Kaggle)", "RGB images, NFT hydroponic lettuce, Raspberry Pi camera",
     f"{len(imgs)} images", "; ".join(f"{k}: {v}" for k, v in counts.items()), "CC0"],
    ["Growth-day sensor data (jurijsruko, Kaggle)", "Daily temperature, humidity, TDS, pH per plant",
     f"{len(sens)} rows / {sens.plant_id.nunique()} plants",
     f"growth day {sens.growth_day.min()}-{sens.growth_day.max()}; {sens.date.min().date()} to {sens.date.max().date()}", "Apache-2.0"],
], columns=["Dataset", "Content", "Size", "Classes / range", "Licence"])
write_table(t1, "table1_datasets", "Datasets used in the case study.", "tab:datasets")

# ---------------- Tables 2 & 3: vision ----------------
oof_p, sum_p = TAB / "cv_vision_oof.csv", TAB / "cv_vision_summary.csv"
if oof_p.exists() and sum_p.exists():
    oof, summ = pd.read_csv(oof_p), pd.read_csv(sum_p)
    classes = sorted(oof.true.unique())
    rng = np.random.RandomState(0)
    rows, recall_rows = [], []
    d0 = {v: oof[(oof.variant == v) & (oof.repeat == 0)].sort_values("filepath").reset_index(drop=True)
          for v in ("unweighted", "weighted")}
    for v, name in [("unweighted", "Standard loss"), ("weighted", "Class-weighted loss")]:
        s = summ[summ.variant == v]
        d = d0[v]
        correct = (d.true == d.pred).values
        boots = [correct[rng.randint(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append([name, mean_sd(s.accuracy), f"[{lo:.3f}, {hi:.3f}]", mean_sd(s.macro_f1), len(d)])
        rec = {}
        for (r, f), g in oof[oof.variant == v].groupby(["repeat", "fold"]):
            for c in classes:
                gc = g[g.true == c]
                if len(gc):
                    rec.setdefault(c, []).append((gc.pred == c).mean())
        recall_rows.append([name] + [mean_sd(rec[c], 2) for c in classes])
    # McNemar (exact) on repeat 0: same images, two models
    a, b = d0["unweighted"], d0["weighted"]
    if len(a) == len(b) and (a.filepath.values == b.filepath.values).all():
        ca, cb = (a.true == a.pred).values, (b.true == b.pred).values
        n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) > 0 else 1.0
        mcn = f"weighted correct / standard wrong = {n01}; standard correct / weighted wrong = {n10}; exact McNemar p = {p:.3f}"
    else:
        mcn = "McNemar test skipped (prediction sets differ)"
    t2 = pd.DataFrame(rows, columns=["Model", "Accuracy (mean +/- SD over folds)", "Accuracy 95% bootstrap CI (pooled)",
                                     "Macro-F1 (mean +/- SD)", "N images tested"])
    write_table(t2, "table2_vision_cv", "Leaf classification under group-aware cross-validation. " + mcn, "tab:vision-cv")
    t3 = pd.DataFrame(recall_rows, columns=["Model"] + [f"{c}" for c in classes])
    write_table(t3, "table3_vision_per_class_recall", "Per-class recall (mean +/- SD across folds).", "tab:vision-recall")
    print("  " + mcn)
else:
    print("  (vision tables skipped: run cv_vision.py first)")

# ---------------- Table 4: sensor ----------------
sp = TAB / "sensor_cv_summary.csv"
if sp.exists():
    s = pd.read_csv(sp)
    s = s.sort_values("r2_mean", ascending=False)
    t4 = pd.DataFrame({
        "Features": s.features, "Model": s.model,
        "R2 (mean +/- SD)": [f"{m:.3f} +/- {sd:.3f}" for m, sd in zip(s.r2_mean, s.r2_sd)],
        "MAE, days (mean +/- SD)": [f"{m:.2f} +/- {sd:.2f}" for m, sd in zip(s.mae_mean, s.mae_sd)],
    })
    write_table(t4, "table4_sensor_cv", "Growth-day prediction under plant-grouped cross-validation (5 folds x 5 repeats); "
                "folds never share a plant. 'date_only' uses calendar date as the sole feature.", "tab:sensor-cv")
else:
    print("  (sensor table skipped: run sensor_analysis.py first)")

print("\nDone. All tables are in paper/outputs/tables (.csv and .tex).")
