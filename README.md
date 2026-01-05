# ISRO Launch Success Predictor

Small Streamlit app that loads ISRO launch data, preprocesses it, trains a RandomForest model, and exposes a UI to predict launch success and view evaluation metrics.

## Features
- CSV loading and cleaning (scripts/preprocess.py)
- One-hot encoding of categorical features
- Model training with optional oversampling (scripts/model.py)
- Streamlit UI for interactive predictions and evaluation (app.py)
- Simple EDA helper (scripts/eda.py)

## Repository layout
- app.py — Streamlit application
- scripts/
  - preprocess.py — load and clean dataset
  - model.py — train_model(X_train, y_train) -> (model, feature_cols)
  - eda.py — small plotting helpers
- data/isro_launches.csv — input dataset (not included)

## Requirements
Recommended packages:
- Python 3.8+
- streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn

Example requirements file is provided.

## Setup (Windows)
1. Create & activate virtual environment
   - PowerShell:
     .\venv\Scripts\Activate.ps1
   - CMD:
     .\venv\Scripts\activate
2. Install dependencies:
   pip install -r requirements.txt

## Run the app
From project root (Windows):
```
cd "c:\ISRO Project"
streamlit run app.py
```

## Notes / Tips
- The app uses caching (st.cache_data / st.cache_resource) to avoid retraining on every rerun.
- The model training function accepts training data only (avoid leakage). If you modify training flow, keep train/test split before training.
- If predict_proba is unavailable, app falls back to predict with a binary mapping.
- Ensure data/isro_launches.csv exists and matches expected columns (`launch_vehicle`, `orbit_type`, `application`, `remarks`/`outcome`).

## Troubleshooting
- Missing imbalanced-learn: `pip install imbalanced-learn`
- Encoding errors loading CSV: try changing encoding or inspect file headers.
