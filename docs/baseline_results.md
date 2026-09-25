# Baseline Model Results — Nutrient Deficiency & Disease Classification

**Model**: MobileNetV2 (ImageNet-pretrained, frozen backbone, fine-tuned classification head)
**Dataset**: Locarno Lettuce Disease Dataset (Rathor, Kaggle) — 209 images, 4 classes
**Split**: 70% train (145) / 15% val (29) / 15% test (35), stratified by class
**Training**: 10 epochs, Adam optimizer, lr=1e-3, CPU

## Class Distribution (Train Set)
| Class | Train Images |
|---|---|
| Healthy | 43 |
| K Deficient | 46 |
| N Deficient | 25 |
| Wilt fungal | 31 |

## Run 1 — Unweighted Cross-Entropy Loss

Peak validation accuracy: 93.1% (epoch 8–9)

**Test set results:**
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Healthy | 1.00 | 1.00 | 1.00 |
| K Deficient | 0.92 | 1.00 | 0.96 |
| N Deficient | 1.00 | 0.67 | 0.80 |
| Wilt fungal | 0.89 | 1.00 | 0.94 |
| **Overall accuracy** | | | **0.94** |

**Finding**: N Deficient recall (67%) was notably weaker than other classes — 2 of 6 test images misclassified. This tracks directly to it being the smallest training class (25 images vs. 43–46 for others), a class imbalance effect rather than a fundamental limitation of the approach.

## Run 2 — Class-Weighted Cross-Entropy Loss

Applied inverse-frequency class weighting to the loss function to counteract the imbalance identified in Run 1.

**Test set results:**
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Healthy | 1.00 | 0.90 | 0.95 |
| K Deficient | 0.91 | 0.91 | 0.91 |
| N Deficient | 0.86 | 1.00 | 0.92 |
| Wilt fungal | 0.88 | 0.88 | 0.88 |
| **Overall accuracy** | | | **0.91** |

## Discussion

Class weighting resolved the N Deficient recall gap entirely (67% → 100%), at the cost of a 3-point drop in overall accuracy (94% → 91%) — a modest redistribution of errors toward the majority classes. This is a deliberate and justifiable trade-off: in a deployed nutrient-monitoring system, failing to detect a genuine nitrogen deficiency (false negative) is more costly than an occasional misclassification among the majority classes, since undetected nutrient stress compounds over time and directly affects crop yield. The class-weighted model is therefore preferred going forward despite its slightly lower raw accuracy.

**Limitation**: Test set size (35 images total, 6–11 per class) is small; these metrics should be treated as indicative rather than statistically robust. Future work includes collecting additional data, particularly for the N Deficient class, to validate this finding at scale.