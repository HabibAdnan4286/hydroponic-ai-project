"""Step 5b - Turn your filled extraction table into the review's tables and figures.

Input : paper/review/extraction.csv  (copy of extraction_template.csv, one row per included study,
        filled using extraction_codebook.md)
Output: table_r1_rigor_indicators, table_r2_task_by_modality, table_r3_quality_scores  (.csv/.tex)
        fig_r2_papers_per_year, fig_r3_modality_task, fig_r4_split_strategy,
        fig_r5_rigor_indicators, fig_r6_dataset_provenance

Every percentage is computed from YOUR extraction - nothing is pre-filled. 'NR' (not reported) is kept as its own
category and counted in the denominator, because under-reporting is itself a finding.

Run:  python paper\\scripts\\analyze_extraction.py
"""
import numpy as np
import pandas as pd

from common import PALETTE, REVIEW, plt, savefig, write_table

path = REVIEW / "extraction.csv"
if not path.exists():
    raise SystemExit("paper/review/extraction.csv not found. Copy extraction_template.csv to extraction.csv and fill it.")
df = pd.read_csv(path, dtype=str).fillna("")
df = df[df.study_id.str.strip() != ""].copy()
n = len(df)
if n < 5:
    raise SystemExit(f"Only {n} studies in extraction.csv - fill in more rows before analysing.")
for c in df.columns:
    df[c] = df[c].str.strip()
print(f"{n} included studies")

YN = ["arid_or_hot_setting", "baselines_compared", "cross_validation", "external_validation",
      "confidence_intervals_or_stat_tests", "explainability", "edge_deployment", "code_or_data_available"]
for c in YN:
    df[c] = df[c].str.upper().replace({"": "NR", "YES": "Y", "NO": "N"})


def split_multi(col):
    return df[col].str.lower().str.replace(" ", "", regex=False).str.split(";").explode().replace("", np.nan).dropna()


# ---- Table R1: rigor indicators ----
ind = []
labels = {
    "baselines_compared": "Compared against baseline models",
    "cross_validation": "Used cross-validation",
    "external_validation": "External validation (other farm / camera / season)",
    "confidence_intervals_or_stat_tests": "Reported confidence intervals or statistical tests",
    "explainability": "Explainability analysis (e.g. Grad-CAM, SHAP)",
    "edge_deployment": "Measured deployment on an embedded device",
    "code_or_data_available": "Code or data available",
    "arid_or_hot_setting": "Arid / hot-climate setting",
}
for c, lab in labels.items():
    k = (df[c] == "Y").sum()
    ind.append([lab, f"{k}", f"{k / n:.0%}"])
ss = df.split_strategy.str.lower().replace("", "not_reported")
ind.append(["Split by plant / unit, time, or site (not random images)", str(ss.isin(["by_plant_or_unit", "temporal", "by_site_or_farm"]).sum()),
            f"{ss.isin(['by_plant_or_unit', 'temporal', 'by_site_or_farm']).mean():.0%}"])
ind.append(["Split strategy not reported", str((ss == "not_reported").sum()), f"{(ss == 'not_reported').mean():.0%}"])
ind.append(["Uses any private (own) dataset", str(df.dataset_source.str.lower().isin(["private", "both"]).sum()),
            f"{df.dataset_source.str.lower().isin(['private', 'both']).mean():.0%}"])
ind.append(["Any vision + sensor fusion", str((df.fusion_type.str.lower().isin(["early", "late", "hybrid"])).sum()),
            f"{df.fusion_type.str.lower().isin(['early', 'late', 'hybrid']).mean():.0%}"])
stress = split_multi("stress_types_covered")
heat_sal = df.stress_types_covered.str.lower().str.contains("heat|salin").sum()
ind.append(["Covers heat or salinity stress", str(int(heat_sal)), f"{heat_sal / n:.0%}"])
r1 = pd.DataFrame(ind, columns=["Indicator", f"Studies (of {n})", "Share"])
write_table(r1, "table_r1_rigor_indicators", f"Methodological indicators across the {n} included studies.", "tab:rigor")
print("\n" + r1.to_string(index=False))

# ---- Table R2: task x modality ----
tm = df[["study_id", "task", "modality"]].copy()
tm["task"] = tm.task.str.lower().str.replace(" ", "").str.split(";")
tm["modality"] = tm.modality.str.lower().str.replace(" ", "").str.split(";")
tm = tm.explode("task").explode("modality").reset_index(drop=True)
tm = tm[(tm.task != "") & (tm.modality != "")].reset_index(drop=True)
ct = pd.crosstab(tm.task, tm.modality)
write_table(ct.reset_index().rename(columns={"task": "Task / modality"}), "table_r2_task_by_modality",
            "Number of studies by task and sensing modality (a study can appear in several cells).", "tab:task-modality")

# ---- Table R3: quality scores ----
qa = [c for c in df.columns if c.startswith("qa")]
score = df[qa].apply(lambda s: s.str.upper().map({"Y": 1, "P": 0.5, "N": 0}).fillna(0)).sum(axis=1)
q = pd.DataFrame({"Quality score (0-6)": score.value_counts().sort_index().index,
                  "Studies": score.value_counts().sort_index().values})
write_table(q, "table_r3_quality_scores", "Distribution of study-quality scores (six-item checklist).", "tab:quality")

# ---- Figures ----
yrs = pd.to_numeric(df.year, errors="coerce").dropna().astype(int).value_counts().sort_index()
fig, ax = plt.subplots(figsize=(4.6, 2.8))
ax.bar(yrs.index.astype(str), yrs.values, color=PALETTE[0])
ax.set_xlabel("Publication year")
ax.set_ylabel("Number of included studies")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
savefig(fig, "fig_r2_papers_per_year")

fig, ax = plt.subplots(figsize=(5.4, 3.6))
im = ax.imshow(ct.values, cmap="Blues")
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        ax.text(j, i, ct.values[i, j], ha="center", va="center", fontsize=8, color="white" if ct.values[i, j] > ct.values.max() / 2 else "black")
ax.set_xticks(range(ct.shape[1]))
ax.set_xticklabels(ct.columns, rotation=40, ha="right")
ax.set_yticks(range(ct.shape[0]))
ax.set_yticklabels(ct.index)
ax.spines[:].set_visible(False)
savefig(fig, "fig_r3_modality_task")

sc = ss.value_counts().reindex(["random_image", "by_plant_or_unit", "temporal", "by_site_or_farm", "not_reported"]).fillna(0)
fig, ax = plt.subplots(figsize=(4.6, 2.8))
ax.barh(["Random image split", "By plant / unit", "Temporal hold-out", "By site / farm", "Not reported"][::-1], sc.values[::-1],
        color=[PALETTE[7], PALETTE[3], PALETTE[2], PALETTE[2], PALETTE[1]][::-1])
ax.set_xlabel("Number of studies")
savefig(fig, "fig_r4_split_strategy")

cols = ["baselines_compared", "cross_validation", "external_validation", "confidence_intervals_or_stat_tests",
        "explainability", "edge_deployment", "code_or_data_available"]
data = np.array([[(df[c] == v).mean() * 100 for v in ("Y", "N", "NR")] for c in cols])
fig, ax = plt.subplots(figsize=(6.2, 3.2))
left = np.zeros(len(cols))
for k, (lab, colr) in enumerate([("Yes", PALETTE[2]), ("No", PALETTE[1]), ("Not reported", "#BBBBBB")]):
    ax.barh(range(len(cols)), data[:, k], left=left, color=colr, label=lab)
    left += data[:, k]
ax.set_yticks(range(len(cols)))
ax.set_yticklabels([labels[c] for c in cols], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Share of included studies (%)")
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.4, 1.0))
savefig(fig, "fig_r5_rigor_indicators")

prov = df.dataset_source.str.lower().replace("", "nr").value_counts()
fig, ax = plt.subplots(figsize=(3.6, 2.8))
ax.bar(prov.index, prov.values, color=PALETTE[0])
ax.set_ylabel("Number of studies")
ax.set_xlabel("Dataset source")
savefig(fig, "fig_r6_dataset_provenance")
print("\nDone.")
