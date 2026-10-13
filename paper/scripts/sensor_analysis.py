"""Step 8a - Sensor case study: confound check + plant-grouped cross-validation + baselines.

Answers, with data, the question you raised: does temperature predict growth day because of biology,
or because temperature simply drifts with the calendar during the experiment?

Checks:
  1. When did each plant start? (all together = strong confound risk; staggered = weaker risk)
  2. Do dates line up with the Growth Days counter? (data integrity)
  3. Feature-set comparison under plant-grouped CV, INCLUDING a calendar-date-only model.
     If date alone predicts growth day as well as temperature does, temperature may be a calendar proxy.
  4. Per-plant Spearman correlation between temperature and growth day (is the sign consistent?)
  5. Permutation importance on held-out plants.

Outputs (paper/outputs/tables): sensor_plant_start_dates.csv, sensor_cv_results.csv, sensor_cv_summary.csv,
  sensor_oof_all4_rf.csv, sensor_perm_importance.csv, sensor_within_plant_corr.csv, sensor_cohort_temp.csv

Run:  python paper\\scripts\\sensor_analysis.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from common import TAB, load_sensor

df = load_sensor()
print(f"Rows: {len(df)}   Plants: {df.plant_id.nunique()}   Growth-day range: {df.growth_day.min()}-{df.growth_day.max()}")
print(f"Calendar range: {df.date.min().date()} to {df.date.max().date()}")

# ---------- 1 & 2: start dates and integrity ----------
df["implied_start"] = df["date"] - pd.to_timedelta(df["growth_day"] - 1, unit="D")
plants = df.groupby("plant_id").agg(start_date=("date", "min"), end_date=("date", "max"), n_days=("growth_day", "count"),
                                    n_implied_starts=("implied_start", "nunique"))
plants["start_month"] = plants.start_date.dt.to_period("M").astype(str)
plants.to_csv(TAB / "sensor_plant_start_dates.csv")
n_unique_starts = plants.start_date.nunique()
start_span = (plants.start_date.max() - plants.start_date.min()).days
print(f"\n[1] Unique plant start dates: {n_unique_starts} of {len(plants)} plants; start dates span {start_span} days")
print("    Plants per start month:\n" + plants.start_month.value_counts().sort_index().to_string())
bad = int((plants.n_implied_starts > 1).sum())
print(f"[2] Plants whose dates do not line up with the Growth Days counter: {bad}")
if n_unique_starts == 1:
    print("    >> WARNING: every plant starts on the same date. Growth Day is then just calendar date minus a constant,")
    print("       so any seasonal temperature trend is fully confounded with growth day.")
elif start_span < 21:
    print("    >> CAUTION: start dates fall in a narrow window; seasonal drift is only weakly de-confounded.")
else:
    print("    >> Start dates are staggered; seasonal drift is partly de-confounded across plants.")

# ---------- 3: grouped CV across feature sets ----------
FEATS = {
    "temp_only": ["temp_c"], "humidity_only": ["humidity"], "tds_only": ["tds"], "ph_only": ["ph"],
    "no_temp (hum+tds+pH)": ["humidity", "tds", "ph"], "all4 (temp+hum+tds+pH)": ["temp_c", "humidity", "tds", "ph"],
    "date_only (calendar)": ["date_index"], "all4 + calendar date": ["temp_c", "humidity", "tds", "ph", "date_index"],
}
PLAN = [  # (feature set, model name)
    ("temp_only", "RF"), ("humidity_only", "RF"), ("tds_only", "RF"), ("ph_only", "RF"),
    ("no_temp (hum+tds+pH)", "RF"), ("all4 (temp+hum+tds+pH)", "RF"), ("all4 (temp+hum+tds+pH)", "Linear"),
    ("all4 (temp+hum+tds+pH)", "GradBoost"), ("date_only (calendar)", "RF"), ("all4 + calendar date", "RF"),
]


def make_model(name, seed):
    if name == "RF":
        return RandomForestRegressor(n_estimators=200, max_depth=10, random_state=seed, n_jobs=-1)
    if name == "GradBoost":
        return GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=seed)
    return LinearRegression()


def plant_folds(ids, k, seed):
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(ids))
    rng.shuffle(ids)
    return np.array_split(ids, k)


K, REPEATS = 5, 5
ids = df.plant_id.unique()
rows, oof_rows, perm_rows = [], [], []
for r in range(REPEATS):
    for f, test_ids in enumerate(plant_folds(ids, K, 1000 + r)):
        te = df.plant_id.isin(test_ids).values
        tr = ~te
        ytr, yte = df.loc[tr, "growth_day"], df.loc[te, "growth_day"]
        dm = DummyRegressor().fit(df.loc[tr, ["temp_c"]], ytr)
        rows.append(dict(repeat=r, fold=f, features="mean baseline", model="Dummy",
                         r2=r2_score(yte, dm.predict(df.loc[te, ["temp_c"]])),
                         mae=mean_absolute_error(yte, dm.predict(df.loc[te, ["temp_c"]]))))
        for fs, mn in PLAN:
            cols = FEATS[fs]
            m = make_model(mn, 7 + r).fit(df.loc[tr, cols], ytr)
            pred = m.predict(df.loc[te, cols])
            rows.append(dict(repeat=r, fold=f, features=fs, model=mn, r2=r2_score(yte, pred), mae=mean_absolute_error(yte, pred)))
            if r == 0 and fs.startswith("all4 (") and mn == "RF":
                o = df.loc[te, ["plant_id", "growth_day", "temp_c", "humidity", "tds", "ph"]].copy()
                o["pred"], o["fold"] = pred, f
                oof_rows.append(o)
                pi = permutation_importance(m, df.loc[te, cols], yte, n_repeats=10, random_state=0, scoring="r2")
                for c, imp in zip(cols, pi.importances_mean):
                    perm_rows.append(dict(fold=f, feature=c, importance=imp))
    print(f"  CV repeat {r + 1}/{REPEATS} done")

res = pd.DataFrame(rows)
res.to_csv(TAB / "sensor_cv_results.csv", index=False)
summ = res.groupby(["features", "model"]).agg(r2_mean=("r2", "mean"), r2_sd=("r2", "std"),
                                              mae_mean=("mae", "mean"), mae_sd=("mae", "std")).reset_index()
summ.to_csv(TAB / "sensor_cv_summary.csv", index=False)
pd.concat(oof_rows).to_csv(TAB / "sensor_oof_all4_rf.csv", index=False)
pd.DataFrame(perm_rows).to_csv(TAB / "sensor_perm_importance.csv", index=False)

print("\n[3] Plant-grouped CV (5 folds x 5 repeats), mean R2 / MAE (days):")
print(summ.sort_values("r2_mean", ascending=False).round(3).to_string(index=False))


def get(fs, mn="RF"):
    return float(summ[(summ.features == fs) & (summ.model == mn)].r2_mean.iloc[0])


t, d = get("temp_only"), get("date_only (calendar)")
print(f"\n    temp_only R2 = {t:.3f}   date_only R2 = {d:.3f}")
if d >= t - 0.02:
    print("    >> Calendar date alone predicts growth day about as well as temperature (or better).")
    print("       Temperature cannot be separated from calendar date in this data, so do not present it as a biological")
    print("       driver without further evidence. Report it as a possible calendar proxy (this is a heuristic, not a proof).")
else:
    print("    >> Temperature predicts better than calendar date alone, which weakens (but does not remove) the confound concern.")

# ---------- 4: within-plant correlation ----------
wc = []
for pid, g in df.groupby("plant_id"):
    if g.temp_c.nunique() > 1:
        rho, p = spearmanr(g.growth_day, g.temp_c)
        wc.append(dict(plant_id=pid, start_month=plants.loc[pid, "start_month"], spearman_rho=rho, p=p))
wc = pd.DataFrame(wc)
wc.to_csv(TAB / "sensor_within_plant_corr.csv", index=False)
print(f"\n[4] Within-plant Spearman(temp, growth day): median {wc.spearman_rho.median():.2f}; "
      f"{(wc.spearman_rho > 0).mean():.0%} of plants positive, {(wc.spearman_rho < 0).mean():.0%} negative")
print("    A mix of signs across start months may indicate that temperature follows the season rather than plant age (heuristic).")

# cohort curves for the figure
df = df.merge(plants[["start_month"]], left_on="plant_id", right_index=True)
df.groupby(["start_month", "growth_day"]).temp_c.mean().reset_index().to_csv(TAB / "sensor_cohort_temp.csv", index=False)

print("\nDone. Next: python paper\\scripts\\make_figures_sensor.py")
