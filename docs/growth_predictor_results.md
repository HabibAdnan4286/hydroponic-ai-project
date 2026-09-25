# Sensor-Based Growth-Day Predictor

**Model**: Random Forest Regressor (200 trees, max depth 10)
**Dataset**: Lettuce Growth Days (Kaggle, jurijsruko) — 3,169 daily readings across 70 plants, 48-day growth window
**Split**: By Plant_ID (49 train / 10 val / 11 test plants) to prevent data leakage across time-series readings of the same plant
**Features**: Temperature (°C), Humidity (%), TDS Value (ppm), pH Level
**Target**: Growth Days (1–48)

## Results
| Metric | Validation | Test |
|---|---|---|
| MAE (days) | 6.01 | 5.78 |
| R² | 0.526 | 0.545 |

## Feature Importance
| Feature | Importance |
|---|---|
| Temperature (°C) | 0.873 |
| TDS Value (ppm) | 0.061 |
| Humidity (%) | 0.043 |
| pH Level | 0.023 |

## Discussion

Temperature dominates as the primary predictor of growth stage, consistent with established plant physiology literature — this has direct relevance to Gulf-region deployments, where ambient heat is a dominant environmental stressor compared to temperate climates. TDS, humidity, and pH contribute comparatively little individually, though this may reflect that they were kept within a narrower controlled range in this dataset's experimental setup rather than being truly unimportant.

**Limitation**: R²≈0.55 indicates roughly half the variance in growth days is unexplained by these four sensor features alone. Likely contributing factors not captured in this dataset include light exposure duration/intensity, nutrient dosing schedule, and individual plant genetic variation. Future work: incorporating light and nutrient-dosing data, and exploring whether combining this sensor-based prediction with the vision-based growth-stage signal (when available) improves overall accuracy — the core motivation for a multi-modal IoT + vision system.