"""Step 7b - All vision figures, from the cross-validation outputs.

Needs: paper/outputs/tables/cv_vision_oof.csv and cv_vision_curves.csv  (from cv_vision.py)
Makes:
  fig_v1_sample_grid        3 example leaves per class
  fig_v2_class_distribution image counts per class
  fig_v3_confusion          row-normalised confusion matrices, unweighted vs class-weighted
  fig_v4_per_class_metrics  precision / recall / F1 per class (mean +/- SD across folds)
  fig_v5_roc                one-vs-rest ROC curves with AUC (class-weighted model)
  fig_v6_training_curves    train loss and held-out accuracy vs epoch (mean +/- SD)

Run:  python paper\\scripts\\make_figures_vision.py
"""
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import auc, confusion_matrix, precision_recall_fscore_support, roc_curve

from common import PALETTE, ROOT, TAB, list_images, plt, savefig

oof_path, curves_path = TAB / "cv_vision_oof.csv", TAB / "cv_vision_curves.csv"
if not oof_path.exists():
    raise SystemExit("cv_vision_oof.csv not found - run cv_vision.py first.")
oof = pd.read_csv(oof_path)
curves = pd.read_csv(curves_path)
imgs = list_images()
classes = sorted(imgs.label.unique())
VAR_ORDER = ["unweighted", "weighted"]
VAR_LABEL = {"unweighted": "Standard loss", "weighted": "Class-weighted loss"}

# ---- V1 sample grid ----
rng = np.random.RandomState(0)
n_show = 3
fig, axes = plt.subplots(len(classes), n_show, figsize=(1.9 * n_show, 1.9 * len(classes)))
for i, c in enumerate(classes):
    pool = imgs[imgs.label == c].filepath.values
    pick = rng.choice(pool, size=min(n_show, len(pool)), replace=False)
    for j in range(n_show):
        ax = axes[i, j]
        ax.axis("off")
        if j < len(pick):
            ax.imshow(Image.open(ROOT / pick[j]).convert("RGB").resize((224, 224)))
        if j == 0:
            ax.set_title(c, loc="left", fontsize=9, fontweight="bold")
savefig(fig, "fig_v1_sample_grid")

# ---- V2 class distribution ----
cnt = imgs.label.value_counts().reindex(classes)
fig, ax = plt.subplots(figsize=(4, 2.6))
bars = ax.bar(classes, cnt.values, color=PALETTE[0])
for b, v in zip(bars, cnt.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.8, str(v), ha="center", fontsize=8)
ax.set_ylabel("Number of images")
ax.set_xlabel("Class")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
savefig(fig, "fig_v2_class_distribution")

# ---- V3 confusion matrices (repeat 0, pooled over folds) ----
fig, axes3 = plt.subplots(1, 3, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [1, 1, 0.05]})
axes, cax = axes3[:2], axes3[2]
for ax, v in zip(axes, VAR_ORDER):
    d = oof[(oof.variant == v) & (oof.repeat == 0)]
    cm = confusion_matrix(d.true, d.pred, labels=classes)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
    for a in range(len(classes)):
        for b in range(len(classes)):
            ax.text(b, a, f"{cm[a, b]}\n({cmn[a, b]:.0%})", ha="center", va="center", fontsize=7,
                    color="white" if cmn[a, b] > 0.5 else "black")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(VAR_LABEL[v])
    ax.spines[:].set_visible(False)
fig.colorbar(im, cax=cax, label="Row-normalised")
savefig(fig, "fig_v3_confusion")

# ---- V4 per-class precision / recall / F1 (per-fold, then mean +/- SD) ----
rows = []
for (v, r, f), d in oof.groupby(["variant", "repeat", "fold"]):
    p, rc, f1, _ = precision_recall_fscore_support(d.true, d.pred, labels=classes, zero_division=0)
    for k, c in enumerate(classes):
        rows.append(dict(variant=v, repeat=r, fold=f, cls=c, precision=p[k], recall=rc[k], f1=f1[k]))
pc = pd.DataFrame(rows)
pc.to_csv(TAB / "cv_vision_per_class_per_fold.csv", index=False)
fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
w = 0.38
for ax, metric in zip(axes, ["precision", "recall", "f1"]):
    for k, v in enumerate(VAR_ORDER):
        g = pc[pc.variant == v].groupby("cls")[metric].agg(["mean", "std"]).reindex(classes).fillna(0)
        ax.bar(np.arange(len(classes)) + (k - 0.5) * w, g["mean"], w, yerr=g["std"], capsize=2,
               color=PALETTE[k], label=VAR_LABEL[v], error_kw=dict(lw=0.8))
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_title(metric.capitalize() if metric != "f1" else "F1-score")
    ax.set_ylim(0, 1.08)
axes[0].set_ylabel("Mean +/- SD across folds")
axes[0].legend(frameon=False, loc="lower left")
savefig(fig, "fig_v4_per_class_metrics")

# ---- V5 ROC one-vs-rest, class-weighted model, repeat 0 ----
d = oof[(oof.variant == "weighted") & (oof.repeat == 0)]
fig, ax = plt.subplots(figsize=(3.8, 3.4))
for k, c in enumerate(classes):
    fpr, tpr, _ = roc_curve((d.true == c).astype(int), d[f"prob_{c}"])
    ax.plot(fpr, tpr, color=PALETTE[k], label=f"{c} (AUC = {auc(fpr, tpr):.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend(frameon=False, loc="lower right", fontsize=7)
savefig(fig, "fig_v5_roc")

# ---- V6 training curves ----
fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
for k, v in enumerate(VAR_ORDER):
    c = curves[curves.variant == v].groupby("epoch")
    for ax, col in zip(axes, ["train_loss", "heldout_acc"]):
        m, s = c[col].mean(), c[col].std().fillna(0)
        ax.plot(m.index, m.values, color=PALETTE[k], label=VAR_LABEL[v])
        ax.fill_between(m.index, m - s, m + s, color=PALETTE[k], alpha=0.2)
axes[0].set_ylabel("Training loss")
axes[1].set_ylabel("Accuracy on held-out fold")
for ax in axes:
    ax.set_xlabel("Epoch")
axes[1].legend(frameon=False)
savefig(fig, "fig_v6_training_curves")

print("\nDone. Next: python paper\\scripts\\gradcam.py")
