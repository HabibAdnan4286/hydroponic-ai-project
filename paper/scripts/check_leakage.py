"""Step 6 - Near-duplicate (leakage) check for the leaf-image dataset.

Why: the dataset has no plant IDs. If several photos show the same plant from nearly the
same angle, a random split can put near-identical images in train AND test, inflating accuracy.

What it does:
  1. Computes a 64-bit perceptual hash (dHash) for every image.
  2. Links images whose Hamming distance <= THRESH into groups (connected components).
  3. Saves the groups (used by cv_vision.py for group-aware cross-validation).
  4. If data/splits.csv exists, measures how close each TEST image is to its nearest TRAIN image.

Limitation (state it in the paper): a perceptual hash only catches visually near-identical
photos (bursts, re-crops). Different views of the same plant will NOT be caught.

Run:  python paper\\scripts\\check_leakage.py [--thresh 10]
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image

from common import ROOT, SPLITS_CSV, TAB, PALETTE, list_images, plt, savefig, write_table

ap = argparse.ArgumentParser()
ap.add_argument("--thresh", type=int, default=10, help="max Hamming distance (of 64 bits) to call two images near-duplicates")
args = ap.parse_args()


def dhash(path, size=8):
    img = Image.open(ROOT / path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


imgs = list_images()
print(f"Hashing {len(imgs)} images ...")
H = np.stack([dhash(p) for p in imgs.filepath])
D = (H[:, None, :] != H[None, :, :]).sum(-1)       # pairwise Hamming distances
np.fill_diagonal(D, 999)

# --- connected components of the "near-duplicate" graph (union-find) ---
parent = list(range(len(imgs)))


def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for i, j in zip(*np.where(np.triu(D <= args.thresh, k=1))):
    parent[find(i)] = find(j)
imgs["group"] = [find(i) for i in range(len(imgs))]
imgs.to_csv(TAB / "leakage_groups.csv", index=False)

sizes = imgs.groupby("group").size()
multi = sizes[sizes > 1]
cross_label = int(sum(imgs[imgs.group == g].label.nunique() > 1 for g in multi.index))
nn_all = D.min(axis=1)

rows = [
    ("Images", len(imgs)),
    ("Hamming threshold (bits of 64)", args.thresh),
    ("Groups (incl. singletons)", int(sizes.shape[0])),
    ("Groups with >1 image", int(multi.shape[0])),
    ("Images inside multi-image groups", int(multi.sum())),
    ("Largest group size", int(sizes.max())),
    ("Multi-image groups spanning >1 class (possible label noise)", cross_label),
    ("Median nearest-neighbour distance, all images", float(np.median(nn_all))),
]

nn_test = None
if SPLITS_CSV.exists():
    sp = pd.read_csv(SPLITS_CSV)
    sp["fp"] = sp["filepath"].astype(str).str.replace("\\", "/", regex=False)
    pos = {fp: i for i, fp in enumerate(imgs.filepath)}
    sp["idx"] = sp["fp"].map(pos)
    if sp["idx"].isna().any():
        print("  WARNING: some paths in data/splits.csv were not found among the images; skipping split check.")
    else:
        tr = sp.loc[sp.split == "train", "idx"].astype(int).values
        te = sp.loc[sp.split == "test", "idx"].astype(int).values
        nn_test = D[np.ix_(te, tr)].min(axis=1)
        rows += [
            ("Test images", len(te)),
            ("Test images with a train image within threshold", int((nn_test <= args.thresh).sum())),
            ("Share of test images within threshold of a train image", f"{(nn_test <= args.thresh).mean():.1%}"),
            ("Median test->train nearest distance", float(np.median(nn_test))),
            ("Min test->train nearest distance", int(nn_test.min())),
        ]
else:
    print("  (data/splits.csv not found - skipping the original-split check)")

summary = pd.DataFrame(rows, columns=["Metric", "Value"])
print("\n" + summary.to_string(index=False))
write_table(summary, "table_leakage_check", "Near-duplicate analysis of the leaf-image dataset (64-bit dHash).", "tab:leakage")

fig, ax = plt.subplots(figsize=(4.2, 2.8))
bins = np.arange(0, 33, 2)
ax.hist(nn_all, bins=bins, color=PALETTE[0], alpha=0.7, label="all images (nearest other image)")
if nn_test is not None:
    ax.hist(nn_test, bins=bins, color=PALETTE[1], alpha=0.7, label="test -> nearest train image")
ax.axvline(args.thresh, color="k", ls="--", lw=1)
ax.text(args.thresh + 0.3, ax.get_ylim()[1] * 0.9, "threshold", fontsize=7)
ax.set_xlabel("Hamming distance between 64-bit perceptual hashes")
ax.set_ylabel("Number of images")
ax.legend(frameon=False)
savefig(fig, "fig_leakage_distance_hist")

print("\nInterpretation guide:")
print("  * Few/no groups with >1 image and test->train distances well above the threshold: no evidence of")
print("    near-duplicate leakage (this does NOT rule out same-plant, different-view leakage).")
print("  * Many multi-image groups or test images within threshold: random splits are optimistic - report")
print("    the group-aware cross-validation results (cv_vision.py uses these groups automatically).")
