# Hydroponic AI: Vision + IoT Prototype and a Validity Case Study

## Motivation
Gulf countries import most of their food because of arid climate and water scarcity; indoor hydroponic farming is one
path toward food security. This repository prototypes machine-learning components for monitoring hydroponic lettuce and
tests, with proper validation, what public datasets can and cannot support.

## What is here
1. **Vision classifier** - MobileNetV2 (transfer learning) classifies lettuce leaves as Healthy, K deficient,
   N deficient or Wilt. 91% accuracy on a 35-image held-out split (see limitations).
2. **Sensor-based growth-day model** - Random Forest on temperature, humidity, TDS and pH, **plus a validity analysis**
   showing that the public dataset cannot support conclusions about temperature effects (all plants share one calendar).
3. **Streamlit demo** (`app/app.py`) for both models.
4. **Paper toolkit** (`paper/`) - scripts for leakage checks, cross-validation, confound analysis, figures and tables.

## Results at a glance
| Component | Result | Caveat |
|---|---|---|
| Leaf classifier | 91% accuracy (class-weighted), single 35-image split | 3 of 35 test images have a near-duplicate in training; group-aware cross-validation is the reportable estimate |
| Growth-day model | R2 0.62 under plant-grouped CV | Calendar date alone gives R2 = 1.00: the model detects the date, not plant biology |

Details: [docs/baseline_results.md](docs/baseline_results.md), [docs/growth_predictor_results.md](docs/growth_predictor_results.md).

## Reproduce
```bash
pip install -r requirements.txt
python -m kaggle datasets download -d rathorhome/lettuce-disease -p data --unzip
python -m kaggle datasets download -d jurijsruko/lettuce -p data/growth-days --unzip
python src/split_data.py && python src/train.py && python src/evaluate.py
python src/split_growth_data.py && python src/train_growth_predictor.py
python paper/scripts/check_leakage.py && python paper/scripts/sensor_analysis.py
python -m streamlit run app/app.py
```

## Limitations
- Both datasets are public, small and single-source; the image data has no plant identifiers.
- The two models are independent. Vision + sensor fusion is proposed, not built.
- The sensor dataset has no staggered planting, so environment effects on growth are not identifiable from it.
- Heat and salinity stress (relevant to arid regions) are not covered by either dataset.

## Next steps
Collect an original dataset with staggered planting, logged environment and images using a low-cost sensor node; build
and evaluate fusion under group-aware, temporal and external validation.

## License
MIT
