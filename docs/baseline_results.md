# Leaf Classifier Results

**Model**: MobileNetV2 (ImageNet-pretrained, frozen backbone, new classification head), Adam lr 1e-3, 10 epochs, CPU
**Dataset**: Lettuce Plant Leaf Disease Dataset (Rathor, Kaggle, CC0), 209 images, 4 classes
(Healthy 62, K Deficient 66, N Deficient 36, Wilt 45)

## Reportable result: 5-fold group-aware cross-validation
Folds are stratified by class and keep near-duplicate images together (`paper/scripts/check_leakage.py`,
`paper/scripts/cv_vision.py`). One repeat; every image is tested once.

| Metric | Standard loss | Class-weighted loss |
|---|---|---|
| Accuracy (mean +/- SD over folds) | 0.842 +/- 0.090 | 0.833 +/- 0.097 |
| Accuracy, pooled 95% bootstrap CI | [0.789, 0.890] | [0.780, 0.880] |
| Macro-F1 | 0.837 +/- 0.084 | 0.829 +/- 0.088 |
| Recall: Healthy | 0.95 +/- 0.07 | 0.94 +/- 0.10 |
| Recall: K Deficient | 0.72 +/- 0.27 | 0.71 +/- 0.28 |
| Recall: N Deficient | 0.86 +/- 0.14 | 0.89 +/- 0.12 |
| Recall: Wilt (fungal) | 0.84 +/- 0.17 | 0.82 +/- 0.15 |

- Class weighting made no significant difference (exact McNemar test: p = 0.625).
- K Deficient is the hardest class. Fold accuracies ranged from 0.67 to 0.91, so small splits are noisy.

## Correction: earlier single-split numbers withdrawn
An earlier version of this page reported 94% (unweighted) and 91% (class-weighted) accuracy on a single 35-image
test split, and a nitrogen-deficiency recall gap (67%) that class weighting appeared to fix (100%). Those figures
rested on six test images in that class and did not replicate: under cross-validation, N Deficient recall is 0.86
even without weighting.

## Leakage check
The dataset has no plant identifiers. A perceptual-hash analysis found 25 groups of near-duplicate images (52 of 209
images), and 3 of the 35 test images in the original split (8.6%) had a near-identical image in training. The hash
check cannot detect different views of the same plant, so some leakage may remain.

## Limitations
Small single-source dataset; one cross-validation repeat (more repeats would tighten the estimates); no external
validation on another farm, camera or season.
