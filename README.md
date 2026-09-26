# Hydroponic AI: Vision + IoT System for Nutrient Stress Detection and Growth Monitoring

## Motivation
Gulf countries import the vast majority of their food due to arid climate and water scarcity, making indoor hydroponic farming a strategic path toward food security. This project prototypes an AI-driven monitoring system for hydroponic lettuce, combining computer vision (leaf health) with IoT sensor data (environmental growth prediction) — a first step toward autonomous, low-labor indoor farms suited to the region's climate and resource constraints.

## What This Project Does
This system has two complementary components:

1. **Vision-based leaf health classifier** — detects nutrient deficiency (Nitrogen, Potassium) and fungal disease (Wilt) from a photograph of a lettuce leaf, using a MobileNetV2 CNN fine-tuned via transfer learning.
2. **Sensor-based growth-day predictor** — estimates a plant's growth stage (day 1–48) from environmental sensor readings (temperature, humidity, TDS, pH) using a Random Forest regressor, identifying temperature as the dominant driver of growth rate — directly relevant to heat-stressed Gulf climates.

Both are wrapped in an interactive Streamlit demo (`app/app.py`) for live testing.

## Results Summary
| Component | Metric | Result |
|---|---|---|
| Leaf health classifier | Test accuracy | 91% (94% before class-balancing fix; see [docs/baseline_results.md](docs/baseline_results.md)) |
| Growth-day predictor | Test R² / MAE | 0.545 / 5.78 days (see [docs/growth_predictor_results.md](docs/growth_predictor_results.md)) |

## Project Structure









## Setup
```bash
pip install -r requirements.txt
```

## Reproducing Results
```bash
# Leaf health classifier
python -m kaggle datasets download -d rathorhome/lettuce-disease -p data --unzip
python src/split_data.py
python src/train.py
python src/evaluate.py

# Growth-day predictor
python -m kaggle datasets download -d jurijsruko/lettuce -p data/growth-days --unzip
python src/split_growth_data.py
python src/train_growth_predictor.py

# Demo app
python -m streamlit run app/app.py
```

## Limitations & Future Work
- **Class imbalance**: Nitrogen-deficiency class remains harder to detect than others due to fewer training samples (see baseline results doc).
- **Growth predictor accuracy**: Environmental sensors alone explain ~55% of growth-day variance; light exposure and nutrient dosing data are likely missing contributors.
- **Models are currently independent.** The core proposed contribution of this line of work is **multi-modal fusion** — combining the vision model's leaf-level stress signal with the sensor model's environmental trend to produce earlier, more reliable stress detection than either signal alone. This project establishes both components as a foundation for that fusion work.
- **Gulf-specific extensions**: incorporating salinity stress and heat stress as explicit classes, using region-representative water and climate conditions, and validating on region-grown cultivars.
- **Edge deployment**: running inference on-device (e.g. Raspberry Pi/Jetson) for low-connectivity farm environments.

## License
MIT