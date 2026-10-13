# Sensor-Based Growth-Day Model: Results and Validity Analysis

**Dataset**: Lettuce Growth Days (Kaggle, jurijsruko), 3,169 daily readings, 70 plants, 48-day window
**Features**: temperature, humidity, TDS, pH. **Target**: Growth Day (1-48)
**Evaluation**: plant-grouped cross-validation, 5 folds x 5 repeats (`paper/scripts/sensor_analysis.py`)

## Correction notice
An earlier version of this page reported temperature as the "dominant driver" of growth (importance 0.87) and linked it
to heat stress in Gulf climates. **That interpretation is withdrawn.** The analysis below shows it was a calendar
confound, not evidence about plant biology.

## What the data shows
All 70 plants started on the same date (2023-08-03), so Growth Day is simply the calendar date minus a constant.

| Features | Model | R2 (mean +/- SD) | MAE (days) |
|---|---|---|---|
| Calendar date only | Random Forest | 1.000 +/- 0.000 | 0.00 |
| Temperature only | Random Forest | 0.633 +/- 0.124 | 4.88 |
| All four sensors | Random Forest | 0.624 +/- 0.118 | 5.00 |
| All four sensors | Gradient boosting | 0.614 +/- 0.121 | 5.35 |
| All four sensors | Linear regression | 0.003 +/- 0.005 | 11.29 |
| Humidity + TDS + pH (no temperature) | Random Forest | -0.038 +/- 0.016 | 11.45 |
| Mean baseline | - | 0.000 +/- 0.000 | 11.32 |

- Calendar date alone predicts growth day perfectly; temperature predicts it only because the temperature trace
  identifies the day within one shared period.
- Within individual plants, temperature and growth day are negatively correlated in 83% of plants (median Spearman
  rho = -0.27) as conditions cooled from August to September - the opposite of a growth-promoting effect of warmth.
- Humidity, TDS and pH carry essentially no information.

## Interpretation
The model is best described as a date detector. This dataset cannot support any claim that temperature (or any other
sensor reading) drives growth stage. Grouped cross-validation by plant does not fix this, because the plants are not
independent experiments: they share one calendar.

## What a valid dataset needs
Staggered planting dates (so environment and plant age vary independently), logged air and root-zone temperature,
light and nutrient dosing, and evaluation on held-out cohorts or seasons. Collecting such data with a low-cost sensor
node is the planned next step.
