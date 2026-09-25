# Hydroponic AI: Growth-Stage & Nutrient-Stress Vision System

## Scope Statement
> TODO: Fill in your one-paragraph scope statement here.
> (What the system does, who it's for, why it matters for Gulf food security.)

## Project Structure
```
data/         # Raw and processed datasets (not committed if large — see .gitignore)
notebooks/    # Exploration, training experiments
src/          # Reusable Python modules (data loading, model, training, inference)
models/       # Saved model weights / checkpoints
app/          # Inference API + demo UI (FastAPI / Streamlit)
docs/         # Architecture diagrams, technical report, dataset provenance notes
```

## Status
🚧 In progress — project scaffolding just created.

## Setup
```bash
pip install -r requirements.txt
```

## Roadmap
- [ ] Download and explore Kaggle Lettuce NPK dataset
- [ ] Train/val/test split
- [ ] Baseline model (transfer learning)
- [ ] Growth-stage recognition head
- [ ] Inference API
- [ ] Demo UI
- [ ] Technical report

## License
MIT
