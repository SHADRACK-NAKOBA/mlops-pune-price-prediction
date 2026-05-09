# Pune Real Estate Price Prediction — MLOps Pipeline

> End-to-end MLOps project: raw Excel data → cleaned features → trained ensemble model → FastAPI service with browser UI. Built as Module 2 of an MLOps course.

[![DVC](https://img.shields.io/badge/pipeline-DVC-945DD6)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [DVC Pipeline](#dvc-pipeline)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [FastAPI Service](#fastapi-service)
- [DagsHub Integration](#dagshub-integration)
- [Configuration](#configuration)
- [Reproducibility](#reproducibility)
- [Tech Stack](#tech-stack)

---

## Overview

This project predicts residential property prices in Pune, India using an ensemble machine learning model. It demonstrates a complete MLOps workflow:

- **Reproducible pipeline** — DVC tracks data, code, and model artifacts through three stages: clean → features → train
- **Experiment tracking** — MLflow logs every run, hyperparameter sweep, and benchmark comparison
- **Production API** — FastAPI serves predictions with 95% confidence intervals
- **Browser UI** — Plain HTML/JS frontend that talks to the API; no build step required
- **AutoML benchmark** — PyCaret benchmarks 20+ algorithms against the hand-tuned ensemble

| Lab | What it produces |
|-----|-----------------|
| Lab 1 | Cleaned dataset (`data_cleaned.csv`) |
| Lab 2 | Feature matrix (`model_features.csv`, `model_target.npy`) + helper artifacts (`model/*.pkl`) |
| Lab 3 | Trained Voting Regressor + 95% interval estimator |
| **Lab 4** | **FastAPI prediction service (`src/`) + browser frontend (`frontend/`)** |
| **Lab 5** | **MLOps tooling: PyCaret / MLflow / DVC / DagsHub (`mlops/`, `dvc.yaml`, `params.yaml`)** |

---

## Architecture

```
Raw Data (Excel)
       │
       ▼
┌──────────────┐     ┌───────────────────┐     ┌─────────────────────────┐
│  Stage 1     │     │  Stage 2          │     │  Stage 3                │
│  clean       │────▶│  features         │────▶│  train                  │
│              │     │                   │     │                         │
│  Validate &  │     │  NLP engineering: │     │  VotingRegressor        │
│  transform   │     │  POS counts,      │     │  (Ridge + Lasso + LR)   │
│  raw Excel   │     │  bigrams,         │     │  + interval estimator   │
│              │     │  target encoding  │     │  + DVC metrics JSON     │
└──────────────┘     └───────────────────┘     └────────────┬────────────┘
                                                            │
                                              ┌─────────────┼──────────────┐
                                              ▼             ▼              ▼
                                         model/*.sav   metrics/        model/*.pkl
                                         (artifacts)  train_metrics   (encoders)
                                                       .json
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  FastAPI          │
                                    │  src/app.py       │
                                    │                   │
                                    │  POST /predict    │
                                    │  GET  /health     │
                                    │  GET  /model/info │
                                    └────────┬──────────┘
                                             │
                                             ▼
                                    frontend/index.html
                                    (Browser UI)
```

---

## Results

Evaluated on a held-out test set (40 samples, 20% split, `random_state=42`):

| Metric | Value |
|--------|-------|
| R² | **0.852** |
| RMSE | **₹15.75 lakhs** |
| MAE | **₹11.46 lakhs** |
| 95% Prediction Interval Margin | ±₹31.35 lakhs |
| Train / Test split | 159 / 40 rows |

### PyCaret AutoML Benchmark

| Model | RMSE (₹ lakhs) | R² |
|-------|----------------|-----|
| Hand-tuned VotingRegressor | 15.75 | 0.852 |
| PyCaret Best Pipeline | 11.79 | 0.917 |

### Ridge Alpha Sweep (MLflow)

| Alpha | R² |
|-------|----|
| 0.01 | ~0.850 |
| 0.1 | ~0.851 |
| **1.0 (default)** | **0.852** |
| **10.0** | **0.854** |
| 100.0 | ~0.851 |
| 1000.0 | ~0.848 |

Run `python -m mlops.mlflow_sweep` then `python -m mlops.mlflow_query` to reproduce this on your machine.

---

## Project Structure

```
mlops-pune-price-prediction/
├── mlops/                          # Training & MLOps scripts (Lab 5)
│   ├── clean_data.py               # Stage 1: data validation & cleaning
│   ├── build_features.py           # Stage 2: NLP feature engineering
│   ├── train.py                    # Stage 3: VotingRegressor training (DVC entry point)
│   ├── utils.py                    # Shared helpers (data, scoring, persistence)
│   ├── mlflow_train.py             # MLflow-tracked training run
│   ├── mlflow_sweep.py             # Ridge alpha hyperparameter sweep
│   ├── mlflow_query.py             # Leaderboard & parallel-coords plot
│   ├── pycaret_benchmark.py        # AutoML comparison (20+ models)
│   ├── dagshub_setup.py            # DagsHub credential & remote setup
│   └── dvc_init.py                 # One-time DVC initialization
├── src/                            # FastAPI inference service (Lab 4)
│   ├── app.py                      # FastAPI app: routes, CORS, startup
│   ├── inference.py                # 7-step prediction pipeline
│   ├── schemas.py                  # Pydantic request/response models
│   └── test_api.py                 # API smoke tests
├── frontend/                       # Browser UI (Lab 4)
│   ├── index.html                  # Input form
│   ├── results.html                # Prediction display
│   ├── script.js                   # Fetch calls to FastAPI
│   └── style.css
├── model/                          # Trained artifacts (DVC-tracked)
│   ├── property_price_prediction_voting.sav  # VotingRegressor
│   ├── interval_est.pkl                      # 95% interval estimator
│   ├── count_vectorizer.pkl                  # Text feature vectorizer
│   ├── sub_area_price_map.pkl                # Target encoding: locality
│   ├── amenities_score_price_map.pkl         # Target encoding: amenities
│   ├── feature_cols.pkl                      # Feature column names
│   └── all_feature_names.pkl                 # Full feature name list
├── metrics/                        # DVC-readable JSON metrics
│   ├── train_metrics.json
│   └── pycaret_benchmark.json
├── dvc.yaml                        # DVC pipeline definition
├── params.yaml                     # Version-controlled hyperparameters
├── requirements.txt                # Production dependencies (Labs 1-4)
├── requirements-mlops.txt          # MLOps tooling: DVC, MLflow, PyCaret (Lab 5)
└── MLOPS_LAB.md                    # Step-by-step Lab 5 guide
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- Git

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd mlops-pune-price-prediction

python -m venv .venv

# Activate virtual environment:
#   Windows PowerShell:  .venv\Scripts\Activate.ps1
#   Windows Cmd:         .venv\Scripts\activate
#   macOS / Linux:       source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the FastAPI Service

```bash
uvicorn src.app:app --reload
```

| URL | Description |
|-----|-------------|
| http://localhost:8000/docs | Interactive Swagger UI |
| http://localhost:8000/health | Liveness check |
| http://localhost:8000/model/info | Model metadata |

### 3. Open the Browser UI

Open `frontend/index.html` directly in your browser. The form posts to `http://localhost:8000/predict` and displays the predicted price with confidence bounds.

### 4. Test the API

```bash
python src/test_api.py
```

---

## DVC Pipeline

Three stages are defined in `dvc.yaml`. Change a value in `params.yaml` and `dvc repro` re-runs only the affected downstream stages.

### Setup & Run

```bash
pip install -r requirements-mlops.txt

# One-time DVC initialization
python -m mlops.dvc_init

# Reproduce all stages (skips unchanged stages automatically)
dvc repro

# View latest metrics
dvc metrics show

# Visualize the dependency graph
dvc dag
```

### Stage Dependencies

| Change | Stages Re-run |
|--------|---------------|
| `Pune Real Estate Data.xlsx` | clean → features → train |
| `mlops/clean_data.py` | clean → features → train |
| `mlops/build_features.py` | features → train |
| `params.yaml` (ridge / lasso / data) | train only |
| `mlops/train.py` | train only |
| `frontend/*` | none |

### Metrics Across Commits

```bash
dvc metrics show          # current commit
dvc metrics diff          # HEAD vs previous commit
dvc metrics diff v1.0     # HEAD vs a tag
```

---

## MLflow Experiment Tracking

All runs are logged to the `M2_Pune_Real_Estate_Price` experiment. The default backend is a local SQLite database (`mlflow.db`).

### Single Training Run

```bash
python -m mlops.mlflow_train
```

Logs: all hyperparameters, RMSE / MAE / R², interval margin, residual diagnostic plots, and model signature.

### Hyperparameter Sweep

```bash
python -m mlops.mlflow_sweep
```

Runs one MLflow run per Ridge alpha in `params.yaml sweep.alphas` (default: `[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]`).

### View Results

```bash
# Print ranked leaderboard to terminal
python -m mlops.mlflow_query

# Launch the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

### AutoML Benchmark

```bash
python -m mlops.pycaret_benchmark
```

Benchmarks 20+ scikit-learn-compatible algorithms via PyCaret. Results saved to `metrics/pycaret_benchmark.json`.

---

## FastAPI Service

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/model/info` | Model type, vocab size, interval margin |
| `POST` | `/predict` | Price prediction with confidence interval |

### Example Request

```json
{
  "property_type": 3,
  "area": 1200.0,
  "sub_area": "kothrud",
  "description": "spacious apartment with modern amenities near park",
  "clubhouse": 1,
  "school": 1,
  "hospital": 0,
  "mall": 1,
  "park": 1,
  "pool": 0,
  "gym": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `property_type` | int | Number of bedrooms (1–10+) |
| `area` | float | Area in square feet |
| `sub_area` | str | Pune locality (e.g. `"kothrud"`, `"baner"`) |
| `description` | str | Free-text property description (optional) |
| `clubhouse` … `gym` | int | Amenity flags — `1` present, `0` absent |

### Example Response

```json
{
  "predicted_price": 85.42,
  "lower_bound": 54.07,
  "upper_bound": 116.77,
  "features_used": 312
}
```

All prices are in **Indian Rupees (₹ lakhs)**.

### Inference Pipeline (`src/inference.py`)

The 7-step pipeline executed per request:

1. **Text cleaning** — lowercase, regex strip, stopword removal (NLTK)
2. **POS tagging** — count nouns, verbs, adjectives from the description
3. **Sub-area encoding** — lookup mean price by Pune locality (fallback to global mean)
4. **Amenity encoding** — lookup mean price by amenity count
5. **Vectorize** — CountVectorizer on the cleaned description
6. **Assemble & predict** — 15 structural features + vocab-size text features → VotingRegressor
7. **Confidence interval** — `margin = z_score × residual_std` from training residuals

---

## DagsHub Integration

[DagsHub](https://dagshub.com) provides hosted MLflow tracking and DVC remote storage for team collaboration.

### Setup

```powershell
# Windows PowerShell
$env:DAGSHUB_USER  = "your_username"
$env:DAGSHUB_TOKEN = "your_token"    # https://dagshub.com/user/settings/tokens
$env:DAGSHUB_REPO  = "your-repo-name"
```

```bash
# macOS / Linux
export DAGSHUB_USER=your_username
export DAGSHUB_TOKEN=your_token
export DAGSHUB_REPO=your-repo-name
```

```bash
# Verify credentials
python -m mlops.dagshub_setup --check

# Point MLflow at DagsHub (current process only)
python -m mlops.dagshub_setup --configure

# Print DVC remote commands (run them to push artifacts to DagsHub)
python -m mlops.dagshub_setup --print-dvc-cmds
```

Once configured, any subsequent `mlflow_train` or `mlflow_sweep` call logs directly to your DagsHub project — no local MLflow UI needed.

---

## Configuration

All tunable parameters live in `params.yaml`. Edit this file and run `dvc repro` — DVC detects which stages are affected.

```yaml
data:
  test_size: 0.2          # Fraction held out for evaluation
  random_state: 42        # Fixed seed for reproducibility

ridge:
  alpha: 1.0              # L2 regularization strength

lasso:
  alpha: 0.1              # L1 regularization strength
  max_iter: 10000         # Wide feature matrix needs extra iterations

voting:
  weights: null           # null = equal weight for LR / Ridge / Lasso

interval:
  confidence: 0.95        # Prediction interval coverage

sweep:
  alphas: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

mlflow:
  experiment_name: M2_Pune_Real_Estate_Price
```

---

## Reproducibility

| Artifact | Versioned by |
|----------|-------------|
| Source code | Git |
| Hyperparameters | `params.yaml` (Git) + MLflow run |
| `model_features.csv`, `model_target.npy` | DVC content hash |
| `model/*.pkl` (encoders) | DVC content hash |
| `model/*.sav` (trained model) | DVC pipeline output |
| Experiment metadata | MLflow run_id |
| Pipeline definition | `dvc.yaml` (Git) |

Full reproduction from scratch:

```bash
git clone <repo>
cd mlops-pune-price-prediction
pip install -r requirements.txt -r requirements-mlops.txt
dvc pull      # fetch data + model from DVC remote
dvc repro     # re-run pipeline → identical numbers
```

> **Note:** The raw Pune Real Estate dataset (`Pune Real Estate Data.xlsx`) is proprietary and not redistributed. You need your own copy to run the `clean` stage from scratch.

---

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Data pipeline | [DVC](https://dvc.org) | Stage orchestration, artifact caching, metric tracking |
| Experiment tracking | [MLflow 3.x](https://mlflow.org) | Run logging, model registry, UI |
| AutoML | [PyCaret 3.x](https://pycaret.org) | Benchmark 20+ algorithms |
| Remote collaboration | [DagsHub](https://dagshub.com) | Hosted MLflow + DVC remote |
| ML models | [scikit-learn](https://scikit-learn.org) | VotingRegressor (Ridge + Lasso + LR) |
| NLP | [NLTK](https://nltk.org) | POS tagging, stopword removal |
| API | [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) | Async REST service |
| Schemas | [Pydantic v2](https://docs.pydantic.dev) | Request/response validation |
| Data | [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) | Preprocessing & feature assembly |
| Config | `params.yaml` + [PyYAML](https://pyyaml.org) | Version-controlled hyperparameters |
| Frontend | HTML / CSS / JavaScript | No-build browser UI |

---

## License

For training and educational use as part of the MLOps course (Module 2).
