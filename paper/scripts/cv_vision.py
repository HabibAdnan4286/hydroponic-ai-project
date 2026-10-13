"""Step 7a - Repeated, group-aware cross-validation of the leaf classifier.

Replaces the single 35-image test split with k-fold CV so every image is tested once per repeat.
Runs BOTH variants from the earlier work (plain vs class-weighted loss) on identical folds, using the
same MobileNetV2 recipe as src/train.py (frozen backbone, new head, Adam 1e-3, batch 8, same augmentation).

Outputs (paper/outputs/tables):
  cv_vision_oof.csv      out-of-fold predictions + class probabilities  (used for confusion matrices, ROC)
  cv_vision_curves.csv   per-epoch train loss and held-out-fold accuracy (used for training curves)
  cv_vision_summary.csv  per-fold accuracy / macro-F1

Run first:  python paper\\scripts\\check_leakage.py   (so folds respect near-duplicate groups)
Then:       python paper\\scripts\\cv_vision.py                 (about 30-60 min on CPU for 1 repeat)
Options:    --epochs 10  --folds 5  --repeats 1  (use --repeats 3 overnight for tighter estimates)

The script saves after every fold, so an interruption does not lose finished folds.
"""
import argparse
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import ROOT, TAB, list_images

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--repeats", type=int, default=1)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

imgs = list_images()
classes = sorted(imgs.label.unique())
c2i = {c: i for i, c in enumerate(classes)}
y = imgs.label.map(c2i).values

groups_csv = TAB / "leakage_groups.csv"
if groups_csv.exists():
    g = pd.read_csv(groups_csv).set_index("filepath")["group"]
    groups = imgs.filepath.map(g)
    if groups.isna().any():
        raise SystemExit("leakage_groups.csv does not match the current images - re-run check_leakage.py")
    groups = groups.astype(int).values
    print(f"Using near-duplicate groups ({len(set(groups))} groups for {len(groups)} images).")
else:
    groups = np.arange(len(imgs))
    print("leakage_groups.csv not found -> plain stratified folds. Run check_leakage.py first for group-aware folds.")

print(f"Classes: {classes}")
print("Pre-loading images ...")
pil = [Image.open(ROOT / fp).convert("RGB").resize((224, 224)) for fp in imgs.filepath]

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), transforms.ToTensor(), norm])
eval_tf = transforms.Compose([transforms.ToTensor(), norm])


class DS(Dataset):
    def __init__(self, idx, tf):
        self.idx, self.tf = idx, tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.tf(pil[j]), int(y[j])


def predict(model, loader):
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            probs.append(torch.softmax(model(xb), dim=1).numpy())
    return np.concatenate(probs)


def run_fold(tr, te, weighted, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    for prm in model.parameters():
        prm.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    counts = np.bincount(y[tr], minlength=len(classes))
    w = torch.tensor(len(tr) / np.maximum(counts, 1), dtype=torch.float32) if weighted else None
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    trl = DataLoader(DS(tr, train_tf), batch_size=8, shuffle=True)
    tel = DataLoader(DS(te, eval_tf), batch_size=16, shuffle=False)
    curves = []
    for ep in range(args.epochs):
        model.train()          # kept identical to src/train.py for comparability
        tot = 0.0
        for xb, yb in trl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item()
        probs = predict(model, tel)
        acc = accuracy_score(y[te], probs.argmax(1))
        curves.append((ep + 1, tot / len(trl), acc))
    return probs, curves


oof_rows, curve_rows, summ_rows = [], [], []
t0 = time.time()
for r in range(args.repeats):
    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed + r)
    for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y, groups)):
        for variant in ("unweighted", "weighted"):
            probs, curves = run_fold(tr, te, variant == "weighted", args.seed + 100 * r + f)
            pred = probs.argmax(1)
            for k, j in enumerate(te):
                row = dict(repeat=r, fold=f, variant=variant, filepath=imgs.filepath[j],
                           true=classes[y[j]], pred=classes[pred[k]])
                row.update({f"prob_{c}": float(probs[k, ci]) for ci, c in enumerate(classes)})
                oof_rows.append(row)
            for ep, loss, acc in curves:
                curve_rows.append(dict(repeat=r, fold=f, variant=variant, epoch=ep, train_loss=loss, heldout_acc=acc))
            summ_rows.append(dict(repeat=r, fold=f, variant=variant, n_test=len(te),
                                  accuracy=accuracy_score(y[te], pred),
                                  macro_f1=f1_score(y[te], pred, average="macro", labels=range(len(classes)), zero_division=0)))
            print(f"[{(time.time() - t0) / 60:5.1f} min] repeat {r} fold {f} {variant:<10} "
                  f"acc={summ_rows[-1]['accuracy']:.3f} macroF1={summ_rows[-1]['macro_f1']:.3f}")
            pd.DataFrame(oof_rows).to_csv(TAB / "cv_vision_oof.csv", index=False)
            pd.DataFrame(curve_rows).to_csv(TAB / "cv_vision_curves.csv", index=False)
            pd.DataFrame(summ_rows).to_csv(TAB / "cv_vision_summary.csv", index=False)

s = pd.DataFrame(summ_rows)
print("\nMean over folds/repeats:")
print(s.groupby("variant")[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(3))
print("\nDone. Next: python paper\\scripts\\make_figures_vision.py")
