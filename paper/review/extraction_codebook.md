# Extraction codebook

Fill **one row per included study** in `extraction.csv` (copy `extraction_template.csv` to `extraction.csv`).
Use ONLY the allowed values below so `analyze_extraction.py` can count them. Use `NR` for "not reported in the paper" -
never guess. If two people extract independently, record Cohen's kappa on a 20% sample.

| Column | Allowed values / format |
|---|---|
| study_id | S001, S002, ... (same ID as in your screening log) |
| first_author, year, venue, doi | free text / 4-digit year |
| crop | lettuce, basil, tomato, strawberry, multiple, other, NR |
| system_type | hydroponic, aquaponic, vertical_farm, greenhouse, other |
| region_or_climate | free text (country or climate zone) |
| arid_or_hot_setting | Y / N / NR (Y only if the paper states an arid, desert, Gulf/GCC or high-heat setting) |
| task | growth_stage, nutrient_deficiency, disease, stress_general, yield_prediction, water_quality, control, other (separate several with `;`) |
| modality | rgb, hyperspectral, multispectral, fluorescence, thermal, sensors, vision+sensors (separate several with `;`) |
| sensors_used | free text (e.g. pH; EC; air temp) |
| stress_types_covered | nutrient, disease, heat, salinity, water, light, none (separate with `;`) |
| dataset_source | public / private / both / NR |
| dataset_name | free text |
| n_samples, n_plants_or_units, n_classes | integers or NR |
| model | free text (e.g. ResNet50; RF; LSTM) |
| baselines_compared | Y / N |
| split_strategy | random_image / by_plant_or_unit / temporal / by_site_or_farm / not_reported |
| cross_validation | Y / N (if Y, put k in notes) |
| external_validation | Y / N (tested on a different farm, camera, cultivar or season) |
| confidence_intervals_or_stat_tests | Y / N |
| explainability | Y / N (Grad-CAM, SHAP, etc.) |
| edge_deployment | Y / N (latency/memory measured on an embedded device) |
| fusion_type | none / early / late / hybrid |
| headline_metric / headline_value | e.g. accuracy / 0.97 |
| code_or_data_available | Y / N |
| qa1 ... qa6 | Y / N / P (partial) - quality checklist, one point each for: clear objective; dataset described; leakage/split addressed; baselines; limitations reported; reproducible |
| notes | free text |

## Suggested search string (a starting point - refine after the scoping search)

```
("hydropon*" OR "indoor farm*" OR "vertical farm*" OR "controlled environment agriculture" OR "aquapon*")
AND ("deep learning" OR "machine learning" OR "computer vision" OR "convolutional" OR "neural network" OR "random forest")
AND ("nutrient" OR "deficien*" OR "growth stage" OR "stress" OR "phenotyp*" OR "disease")
```

Databases: Scopus, Web of Science, IEEE Xplore, ScienceDirect (+ Google Scholar for citation chasing).
Date range suggestion: 2018 to the search date. Record the **exact search date and the exact string per database** - reviewers ask for this.

## Exclusion codes (match `config_prisma.json`)

E1 not hydroponic / controlled-environment, E2 no machine-learning method, E3 no empirical evaluation,
E4 not peer-reviewed or no full text, E5 other (state the reason in notes).
