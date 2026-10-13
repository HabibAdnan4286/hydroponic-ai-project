"""Shared helpers for the paper toolkit.

All scripts are meant to be run from the project root, e.g.
    python paper\\scripts\\check_leakage.py
Paths are resolved relative to this file, so the working directory does not matter.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # project root
PAPER = ROOT / "paper"
FIG = PAPER / "outputs" / "figures"
TAB = PAPER / "outputs" / "tables"
REVIEW = PAPER / "review"
for _d in (FIG, TAB, REVIEW):
    _d.mkdir(parents=True, exist_ok=True)

IMG_DIR = ROOT / "data" / "Lettuce disease"
SENSOR_CSV = ROOT / "data" / "growth-days" / "lettuce_dataset_updated.csv"
SPLITS_CSV = ROOT / "data" / "splits.csv"

# Okabe-Ito colour-blind-safe palette
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def savefig(fig, name):
    """Save as PNG (300 dpi) and PDF (vector) into paper/outputs/figures."""
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure: {name}.png / .pdf")


def list_images():
    """Return DataFrame(filepath, label) for every image in data/Lettuce disease/<class>/.
    filepath is stored relative to the project root with forward slashes."""
    if not IMG_DIR.exists():
        raise SystemExit(f"Image folder not found: {IMG_DIR}")
    rows = []
    for cls_dir in sorted(p for p in IMG_DIR.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append((f.relative_to(ROOT).as_posix(), cls_dir.name))
    if not rows:
        raise SystemExit(f"No images found under {IMG_DIR}")
    return pd.DataFrame(rows, columns=["filepath", "label"])


SENSOR_COLS = ["plant_id", "date", "temp_c", "humidity", "tds", "ph",
               "growth_day", "temp_f", "humidity_frac"]


def load_sensor():
    """Load the Kaggle 'Lettuce Growth Days' CSV with clean column names.

    The file is latin-1 encoded (degree symbol). Columns are renamed by position,
    and the two redundant unit-conversion columns (temp_f, humidity_frac) are dropped.
    """
    if not SENSOR_CSV.exists():
        raise SystemExit(f"Sensor CSV not found: {SENSOR_CSV}")
    df = pd.read_csv(SENSOR_CSV, encoding="latin1")
    if df.shape[1] != 9:
        raise SystemExit(f"Expected 9 columns in sensor CSV, found {df.shape[1]}: {list(df.columns)}")
    df.columns = SENSOR_COLS
    df = df.drop(columns=["temp_f", "humidity_frac"])
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    n_bad = int(df["date"].isna().sum())
    if n_bad:
        print(f"  WARNING: {n_bad} rows have unparseable dates (expected format M/D/YYYY) and will be dropped.")
        df = df.dropna(subset=["date"])
    df["date_index"] = (df["date"] - df["date"].min()).dt.days   # days since earliest date in dataset
    return df.sort_values(["plant_id", "growth_day"]).reset_index(drop=True)


def _tex_escape(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def write_table(df, name, caption, label, index=False):
    """Write a table as CSV and as a booktabs LaTeX file (no extra dependencies)."""
    df = df.reset_index() if index else df
    df.to_csv(TAB / f"{name}.csv", index=False)
    cols = list(df.columns)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{_tex_escape(caption)}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "l" + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(_tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            sv = str(v)
            cells.append(sv.replace("+/-", r"$\pm$") if "+/-" in sv else _tex_escape(sv))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved table: {name}.csv / .tex")


def mean_sd(x, digits=3):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.{digits}f} +/- {x.std(ddof=1) if len(x) > 1 else 0.0:.{digits}f}"
