# Paper toolkit - systematic review + empirical case study

**Working title (pick one):**
1. *How rigorous is the evidence? A systematic review of vision and IoT-sensor machine learning for hydroponic crop monitoring, with a case study*
2. *Validity pitfalls in AI for hydroponic crop monitoring: a systematic review and a reproducible case study*

**Decision (see the chat for the reasoning):** methodological-rigor systematic review as the main contribution, with
multimodal vision+sensor fusion as the identified gap and hot/arid-climate stress as the motivating context. The
prototype is a *small, clearly-labelled case study* that demonstrates the pitfalls the review documents. It is NOT the
paper's main evidence.

**Research questions**
- RQ1 What tasks, modalities and models are used for ML-based monitoring in hydroponic / controlled-environment farming?
- RQ2 How are these models validated (split strategy, cross-validation, external validation, uncertainty, baselines)?
- RQ3 How often are vision and sensor data fused, and with what benefit?
- RQ4 How well are hot/arid-climate stresses (heat, salinity) covered?

**Go / no-go rule:** run the scoping search first. If fewer than ~40 studies meet the criteria, write a *scoping* review
instead of a systematic one. If the methodological angle turns out to be already covered by a recent review, shift the
emphasis to fusion (RQ3).

---

## Run order (from the project root, Windows)

| Step | Command | Produces |
|---|---|---|
| 1 | `pip install matplotlib scipy` (if missing) | |
| 2 | `python paper\scripts\check_leakage.py` | near-duplicate groups, leakage table + figure |
| 3 | `python paper\scripts\cv_vision.py` (30-60 min; `--repeats 3` overnight) | out-of-fold predictions, curves |
| 4 | `python paper\scripts\make_figures_vision.py` | fig V1-V6 |
| 5 | `python paper\scripts\gradcam.py` | fig V7-V8 |
| 6 | `python paper\scripts\sensor_analysis.py` | confound check, grouped CV, baselines |
| 7 | `python paper\scripts\make_figures_sensor.py` | fig S1-S6 |
| 8 | `python paper\scripts\make_tables.py` | tables 1-4 (CSV + LaTeX) |
| 9 | `python paper\scripts\make_conceptual_figures.py` | fig C1 (framework), C2 (validation ladder) |
| 10 | fill `paper\review\extraction.csv`, then `python paper\scripts\analyze_extraction.py` | review tables R1-R3, fig R2-R6 |
| 11 | fill `paper\review\config_prisma.json`, then `python paper\scripts\make_prisma.py` | fig R1 (PRISMA) |

Everything lands in `paper\outputs\figures` (PNG 300 dpi + vector PDF) and `paper\outputs\tables` (CSV + LaTeX).

## What was tested, and what was not
- Tested on synthetic data in a sandbox: check_leakage, sensor_analysis, make_figures_sensor, make_figures_vision,
  make_tables, make_conceptual_figures, make_prisma (including its refusal on empty/inconsistent counts),
  analyze_extraction. Two layout bugs and one pandas bug found this way were fixed.
- **NOT run (no PyTorch in the sandbox): `cv_vision.py` and `gradcam.py`** - syntax-checked only. Expect to debug
  small things on first run; paste any error back and it will be fixed.
- **No numbers in this kit come from the literature or from your data.** Every result appears only after you run the
  scripts. The review tables are computed from YOUR extraction; PRISMA counts come from YOUR screening log.

## Figure / table map (where each goes in the paper)

| Section | Figures | Tables |
|---|---|---|
| 1 Introduction | C1 framework (motivation) | |
| 2 Methods: review protocol | R1 PRISMA | codebook (supplement) |
| 3 Results: landscape | R2 papers per year, R3 task x modality, R6 dataset provenance | R2 task by modality |
| 4 Results: validation rigor (core) | R4 split strategy, R5 rigor indicators, C2 validation ladder | R1 rigor indicators, R3 quality scores |
| 5 Case study: data | V1 samples, V2 class counts, S5 time series, V9 leakage histogram | 1 datasets, leakage table |
| 5 Case study: vision | V3 confusion, V4 per-class, V5 ROC, V6 curves, V7-V8 Grad-CAM | 2 CV results, 3 per-class recall |
| 5 Case study: sensors | S1 pred vs actual, S2 residuals, S3 importance, S4 correlation, **S6 confound** | 4 sensor CV |
| 6 Discussion: fusion gap, hot-climate gap | C1 framework | |
| 7 Limitations and future work | | |

(The leakage histogram file is `fig_leakage_distance_hist`.)

## Honest limitations to write into the paper
- Both datasets are public and small; the vision data has no plant IDs, so same-plant leakage can only be partly checked.
- Whether temperature reflects biology or the calendar depends on the confound check - report whatever `sensor_analysis.py` finds.
- No fusion model has been built; the framework figure marks fusion as proposed.
- Journal quartiles change yearly: check SCImago / JCR for the current year before choosing a venue.
