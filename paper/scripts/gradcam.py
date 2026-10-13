"""Step 7c - Grad-CAM heatmaps: where does the CNN look?

Reviewers in plant science nearly always ask whether the network looks at the leaf (symptom) or at the
background / pot / label. This script overlays Grad-CAM on test images using the model you already trained
(models/baseline_mobilenetv2.pth, the class-weighted run) and the test split in data/splits.csv.

Makes:
  fig_v7_gradcam_correct        up to 3 correctly classified test images per class
  fig_v8_gradcam_errors         misclassified test images (error analysis), if any
  table_gradcam_predictions.csv per-image prediction, confidence, correct/incorrect

IMPORTANT for the paper: look at every heatmap yourself and describe honestly what you see. If the
model attends to background, say so - that is a finding, not a failure.

Run:  python paper\\scripts\\gradcam.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from common import ROOT, SPLITS_CSV, TAB, list_images, plt, savefig

WEIGHTS = ROOT / "models" / "baseline_mobilenetv2.pth"
if not WEIGHTS.exists() or not SPLITS_CSV.exists():
    raise SystemExit("Need models/baseline_mobilenetv2.pth and data/splits.csv (from the earlier steps).")

classes = sorted(list_images().label.unique())
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.eval()

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
to_tensor = transforms.Compose([transforms.ToTensor(), norm])

store = {}
layer = model.features[-1]
layer.register_forward_hook(lambda m, i, o: (store.__setitem__("act", o), o.register_hook(lambda g: store.__setitem__("grad", g))))


def gradcam(pil_img):
    x = to_tensor(pil_img).unsqueeze(0)
    out = model(x)
    probs = torch.softmax(out, 1)[0].detach().numpy()
    cls = int(out.argmax(1))
    model.zero_grad()
    out[0, cls].backward()
    w = store["grad"].mean(dim=(2, 3), keepdim=True)                 # channel weights
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))            # 1 x 1 x 7 x 7
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, cls, probs


sp = pd.read_csv(SPLITS_CSV)
sp = sp[sp.split == "test"].reset_index(drop=True)
records = []
for _, r in sp.iterrows():
    fp = str(r["filepath"]).replace("\\", "/")
    pil = Image.open(ROOT / fp).convert("RGB").resize((224, 224))
    cam, cls, probs = gradcam(pil)
    records.append(dict(filepath=fp, true=r["label"], pred=classes[cls], conf=float(probs[cls]), pil=pil, cam=cam))

pd.DataFrame([{k: v for k, v in d.items() if k not in ("pil", "cam")} for d in records]).to_csv(
    TAB / "table_gradcam_predictions.csv", index=False)


def draw(ax, d):
    ax.imshow(d["pil"])
    ax.imshow(d["cam"], cmap="jet", alpha=0.45)
    ax.axis("off")
    ok = d["true"] == d["pred"]
    ax.set_title(f"true: {d['true']}\npred: {d['pred']} ({d['conf']:.0%})", fontsize=7, color="black" if ok else "#D55E00")


# correct predictions, up to 3 per class
n = 3
fig, axes = plt.subplots(len(classes), n, figsize=(2.1 * n, 2.3 * len(classes)), squeeze=False)
for i, c in enumerate(classes):
    good = [d for d in records if d["true"] == c and d["pred"] == c][:n]
    for j in range(n):
        axes[i, j].axis("off")
        if j < len(good):
            draw(axes[i, j], good[j])
savefig(fig, "fig_v7_gradcam_correct")

# errors
errs = [d for d in records if d["true"] != d["pred"]]
if errs:
    cols = min(4, len(errs))
    rows = int(np.ceil(len(errs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, d in zip(axes.ravel(), errs):
        draw(ax, d)
    savefig(fig, "fig_v8_gradcam_errors")
else:
    print("  No misclassified test images - no error gallery made.")

print(f"\n{len(errs)} of {len(records)} test images misclassified. Write down, for each error, your hypothesis for why.")
