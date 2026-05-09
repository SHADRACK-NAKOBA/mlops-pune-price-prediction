# Pune Real Estate Price Prediction — MLOps Pipeline

> End-to-end MLOps project: raw Excel data → cleaned features → trained ensemble model → FastAPI service with browser UI. Built as Module 2 of an MLOps course.

[![DVC](https://img.shields.io/badge/pipeline-DVC-945DD6)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)

---

## Table of Contents

- [Project Scope](#project-scope)
- [Architecture](#architecture)
- [Results](#results)
- [Project Structure — What Each File Does](#project-structure--what-each-file-does)
- [How the ML Pipeline Works](#how-the-ml-pipeline-works)
- [How the API Works](#how-the-api-works)
- [How Experiment Tracking Works](#how-experiment-tracking-works)
- [Quick Start](#quick-start)
- [DVC Pipeline](#dvc-pipeline)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [FastAPI Service](#fastapi-service)
- [DagsHub Integration](#dagshub-integration)
- [Configuration](#configuration)
- [Reproducibility](#reproducibility)
- [Tech Stack](#tech-stack)

---

## Project Scope

### What this project is

This is a **production-style MLOps project** built on a real-world dataset of ~200 residential property listings in Pune, India. The goal is to predict the listed price of a property (in ₹ lakhs) given its size, location, amenities, and a text description.

The project is structured as five progressive labs, each building on the last:

| Lab | Focus | What it produces |
|-----|-------|-----------------|
| Lab 1 | Data cleaning | `data_cleaned.csv` — validated, standardised property records |
| Lab 2 | Feature engineering | `model_features.csv` — NLP-enriched feature matrix ready for training |
| Lab 3 | Model training | Trained VotingRegressor + 95% prediction interval estimator |
| Lab 4 | Serving | FastAPI REST service + plain HTML/JS browser UI |
| Lab 5 | MLOps | DVC pipeline, MLflow experiment tracking, PyCaret benchmark, DagsHub integration |

The project demonstrates how the same model can be developed iteratively (notebooks / scripts), tracked rigorously (MLflow, DVC), served reliably (FastAPI), and compared against AutoML alternatives (PyCaret) — all without locking you into a heavy platform.

### What this project is not

- It is **not a deep learning project**. The model is a classical ensemble of three linear regressors (Ridge, Lasso, ordinary Linear Regression). This is intentional — the dataset is small (~200 rows) and linear models generalise better at this scale.
- It is **not a production deployment**. There is no Docker image, no cloud infrastructure, and no authentication on the API. The FastAPI server is designed to run locally.
- It is **not a data collection tool**. The raw dataset (`Pune Real Estate Data.xlsx`) is proprietary and not redistributed here. You need your own copy to run the pipeline from scratch.

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
                                         (trained     train_metrics   (encoders &
                                          model)       .json           vectorizer)
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

**DVC** orchestrates the three training stages and caches every artifact — if nothing changed, it skips the stage. **MLflow** sits alongside the pipeline for experiment tracking; it records what each training run produced without being part of the DVC DAG itself. The **FastAPI service** loads the trained artifacts at startup and serves predictions; it has no coupling to DVC or MLflow at runtime.

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

The PyCaret pipeline outperforms the manual ensemble by about 6.5% in R². This is expected — PyCaret searches over many algorithms and applies cross-validated stacking. The hand-tuned model is simpler, more interpretable, and serves as a solid baseline.

### Ridge Alpha Sweep (MLflow)

| Alpha | R² |
|-------|----|
| 0.01 | ~0.850 |
| 0.1 | ~0.851 |
| **1.0 (default)** | **0.852** |
| **10.0** | **0.854** |
| 100.0 | ~0.851 |
| 1000.0 | ~0.848 |

The default alpha of 1.0 is close to optimal. Increasing to 10.0 gives a marginal improvement (ΔR² ≈ 0.002). Run `python -m mlops.mlflow_sweep` then `python -m mlops.mlflow_query` to reproduce this.

---

## Project Structure — What Each File Does

```
mlops-pune-price-prediction/
├── mlops/                   ← All training & MLOps scripts (Lab 5)
├── src/                     ← FastAPI service (Lab 4)
├── frontend/                ← Browser UI (Lab 4)
├── model/                   ← Trained artifacts produced by the pipeline
├── metrics/                 ← JSON metric files DVC can read and diff
├── dvc.yaml                 ← Declares the three pipeline stages
├── params.yaml              ← Single source of truth for hyperparameters
├── requirements.txt         ← Production dependencies
└── requirements-mlops.txt  ← MLOps-only dependencies (DVC, MLflow, PyCaret)
```

### `mlops/` — Training & MLOps Scripts

| File | What it does |
|------|-------------|
| `clean_data.py` | **Stage 1 of the DVC pipeline.** Reads the raw Excel file, drops malformed rows, standardises column names, converts price to ₹ lakhs, and writes `data_cleaned.csv`. |
| `build_features.py` | **Stage 2.** Builds the full feature matrix. Cleans the description text (lowercase, strip punctuation, remove stopwords), runs NLTK POS tagging to count nouns/verbs/adjectives, applies target encoding for `sub_area` and `amenity_score`, and runs a CountVectorizer on the description. Writes `model_features.csv`, `model_target.npy`, and five `.pkl` encoder artifacts that the inference service needs at runtime. |
| `train.py` | **Stage 3.** Loads the feature matrix, splits train/test using params from `params.yaml`, builds a `VotingRegressor` (Ridge + Lasso + LinearRegression with equal weights), fits it, computes an interval estimator from training residuals, and writes the model, interval estimator, and `metrics/train_metrics.json`. This is the DVC entry point — it has no MLflow calls so DVC can run it cleanly. |
| `utils.py` | **Shared utilities** used by every other script. Contains `load_params()` (reads `params.yaml`), `load_data()`, `build_model()`, `score_model()`, `estimate_interval()`, and `save_artifacts()`. Centralising these means a change to the scoring logic is reflected everywhere automatically. |
| `mlflow_train.py` | Runs the same training as `train.py` but wraps it in an MLflow run. Logs all params, RMSE/MAE/R², interval margin, a residuals-vs-predicted diagnostic plot, and the model with a signature (so MLflow can later serve it with `mlflow models serve`). Tags the run as a candidate for the model registry. |
| `mlflow_sweep.py` | Iterates over every alpha in `params.yaml sweep.alphas`, training and logging one MLflow run per value. Only Ridge alpha varies — Lasso and LR stay at their defaults. Useful for understanding how sensitive the model is to regularisation strength. |
| `mlflow_query.py` | Queries all runs in the experiment via the MLflow tracking API, prints a ranked leaderboard sorted by R², and generates a parallel-coordinates plot showing how each hyperparameter configuration relates to model performance. |
| `pycaret_benchmark.py` | Uses PyCaret's `compare_models()` to evaluate 20+ scikit-learn-compatible algorithms (Random Forest, XGBoost, LightGBM, CatBoost, etc.) on the same train/test split. Saves the top results to `metrics/pycaret_benchmark.json` so DVC can track them alongside the manual model. |
| `dagshub_setup.py` | A helper that reads three environment variables (`DAGSHUB_USER`, `DAGSHUB_TOKEN`, `DAGSHUB_REPO`) and either verifies them (`--check`), sets the MLflow tracking URI to DagsHub (`--configure`), or prints the exact `dvc remote add` commands needed to point DVC at DagsHub (`--print-dvc-cmds`). It never modifies shell config files — it only affects the current Python process. |
| `dvc_init.py` | One-time DVC setup: runs `dvc init`, adds the pipeline stages if they are not already in `dvc.yaml`, and stages the DVC config for the first Git commit. Only needs to be run once per clone. |

### `src/` — FastAPI Inference Service

| File | What it does |
|------|-------------|
| `app.py` | The FastAPI application. Defines three routes (`GET /health`, `GET /model/info`, `POST /predict`), enables CORS for all origins (so the local HTML frontend can call it), and loads all model artifacts once at startup via `inference.py`. |
| `inference.py` | The prediction engine. At startup it loads six artifacts from `model/`: the VotingRegressor, the CountVectorizer, two target-encoding maps (sub-area and amenity score), the feature column names, and the interval estimator. The `predict_price()` function runs a 7-step pipeline (text cleaning → POS tagging → target encoding → vectorisation → feature assembly → prediction → confidence interval) and returns a dict with the point estimate and bounds. |
| `schemas.py` | Pydantic v2 models for the API. `PropertyInput` defines and validates the request body (11 fields). `PriceResponse` defines the response (predicted price, lower bound, upper bound, number of features used). `HealthResponse` and `ModelInfoResponse` cover the other two endpoints. Pydantic rejects bad input automatically with a clear 422 error. |
| `test_api.py` | A simple HTTP client test script. Sends a hard-coded `PropertyInput` to the running server and prints the response. Useful for a quick smoke test after starting the server. Not a unit test suite — just a sanity check. |

### `frontend/` — Browser UI

| File | What it does |
|------|-------------|
| `index.html` | The input form. Has fields for all 11 `PropertyInput` parameters. On submit, `script.js` sends a `POST /predict` request to the FastAPI server and redirects to `results.html`. |
| `results.html` | Displays the prediction. Shows the point estimate, the lower/upper confidence bounds, and the number of features the model used. |
| `script.js` | The glue. Serialises the form into a JSON body, calls `fetch()` against `http://localhost:8000/predict`, and passes the result to the results page via URL query params. |
| `style.css` | Minimal styling. No framework dependencies. |

### `model/` — Trained Artifacts

These files are produced by the DVC pipeline and loaded by the FastAPI service at startup. They are tracked by DVC (not committed to Git directly).

| File | What it contains |
|------|-----------------|
| `property_price_prediction_voting.sav` | The fitted `VotingRegressor`. Serialised with `joblib`. |
| `interval_est.pkl` | A dict with two keys: `z_score` (from the confidence level in `params.yaml`) and `residual_std` (standard deviation of training residuals). At inference time, `margin = z_score × residual_std`. |
| `count_vectorizer.pkl` | The fitted `CountVectorizer`. Transforms the property description into a sparse bag-of-words matrix. The vocabulary is fixed at training time — unknown words at inference are silently ignored. |
| `sub_area_price_map.pkl` | A dict mapping each Pune locality (e.g. `"kothrud"`, `"baner"`) to the mean property price in that area, computed from the training set. This is **target encoding** — it replaces a categorical variable with a numeric signal derived from the label. |
| `amenities_score_price_map.pkl` | Similar target encoding but for amenity count (0–7). Maps the number of amenities a property has to the mean price of training properties with that amenity count. |
| `feature_cols.pkl` | The list of structural feature column names (15 features: `property_type`, `area`, 7 amenity flags, and 4 derived numeric features). Used to build the DataFrame in the correct column order at inference time. |
| `all_feature_names.pkl` | The complete ordered list of all feature names (structural + vectorizer vocabulary). Used to create the named DataFrame that the VotingRegressor expects. |

### `metrics/` — DVC Metric Files

| File | What it contains |
|------|-----------------|
| `train_metrics.json` | RMSE, MAE, and R² from the most recent `dvc repro` run. Committed to Git (with `cache: false` in `dvc.yaml`) so `dvc metrics diff` can compare across commits. |
| `pycaret_benchmark.json` | Top algorithm results from `pycaret_benchmark.py`. Stored alongside the manual model metrics for easy comparison. |

### Root Config Files

| File | What it does |
|------|-------------|
| `dvc.yaml` | Defines the three pipeline stages (`clean`, `features`, `train`), their commands, dependencies, parameters, and outputs. DVC reads this to build the DAG and decide which stages to re-run. |
| `params.yaml` | All hyperparameters in one place. DVC watches this file — if a value changes, the affected stage re-runs. Code reads it via `mlops.utils.load_params()`. |
| `requirements.txt` | Dependencies for Labs 1–4: FastAPI, uvicorn, scikit-learn, NLTK, pandas, NumPy, scipy, matplotlib, seaborn. Install this for the API service. |
| `requirements-mlops.txt` | Additional dependencies for Lab 5 only: MLflow, DVC, DagsHub, PyCaret. Kept separate to avoid inflating the API service's dependency footprint. |
| `MLOPS_LAB.md` | Step-by-step guide for completing Lab 5, including exact commands for each tool. |
| `.gitignore` | Excludes virtual environments, Python bytecode, the raw Excel file, DVC cache, MLflow store, NLTK downloads, and OS/IDE artefacts. |
| `.dvcignore` | Tells DVC which files to ignore when computing content hashes (similar to `.gitignore` but for DVC). |

---

## How the ML Pipeline Works

### Stage 1 — Data Cleaning (`mlops/clean_data.py`)

The raw Excel sheet has inconsistent formatting: price columns may contain strings like `"85 Lakhs"` or `"1.2 Cr"`, area is sometimes in sq ft and sometimes in sq m, and some rows are missing key fields. `clean_data.py` standardises all of this:

- Converts all prices to ₹ lakhs (crore values are multiplied by 100)
- Normalises area to square feet
- Drops rows with null values in price, area, or sub_area
- Lowercases and strips whitespace from text fields
- Writes the clean result to `data_cleaned.csv`

### Stage 2 — Feature Engineering (`mlops/build_features.py`)

This stage turns the clean tabular data into a numeric feature matrix the model can use. It has three main parts:

**Structural features (15 columns)**

| Feature | Type | Description |
|---------|------|-------------|
| `property_type` | int | Number of bedrooms |
| `area` | float | Area in sq ft |
| `clubhouse`, `school`, `hospital`, `mall`, `park`, `pool`, `gym` | int (0/1) | Amenity presence flags |
| `price_by_subarea` | float | Target-encoded locality mean price |
| `amenity_score` | int | Sum of the 7 amenity flags (0–7) |
| `price_by_amenities` | float | Target-encoded amenity-count mean price |
| `noun_count`, `verb_count`, `adj_count` | int | POS tag counts from the description |

**Target encoding** replaces high-cardinality categorical columns (locality, amenity count) with the mean target value for that category, computed on the training split. This lets the model use location and amenity richness as a price signal without one-hot encoding dozens of dummy columns.

**NLP features (vocabulary-size columns)**

The property description is cleaned (lowercase, strip punctuation, remove stopwords), then transformed by a `CountVectorizer` into a bag-of-words matrix. Each column represents one word from the training vocabulary; the cell value is how many times that word appears in the description. This lets the model pick up on words like `"spacious"`, `"renovated"`, or `"sea view"` that correlate with price.

**POS tagging** (Part-of-Speech tagging using NLTK) counts nouns, verbs, and adjectives in the description separately. Listings with many adjectives tend to be higher-quality descriptions of higher-value properties.

### Stage 3 — Training (`mlops/train.py`)

The model is a `VotingRegressor` combining three estimators with equal weight:

- **Ridge Regression** — linear regression with L2 regularisation. Shrinks all coefficients towards zero, reducing overfitting on the high-dimensional text features.
- **Lasso Regression** — linear regression with L1 regularisation. Drives some coefficients to exactly zero, effectively selecting a sparse subset of features.
- **Linear Regression** — ordinary least squares, unregularised. Acts as an anchor.

The ensemble averages the three predictions. Each estimator has different biases and the average tends to be more stable than any single one.

**Prediction interval estimation** — after fitting, the model predicts on the training set and computes the standard deviation of the residuals (`residual_std`). At inference time, the 95% interval is `prediction ± z_score × residual_std`, where `z_score = 1.96` for a 95% normal interval. This is a simple parametric interval — it assumes the residuals are roughly normally distributed and homoscedastic (constant variance). For a dataset of 200 rows it is a reasonable approximation.

---

## How the API Works

The FastAPI service (`src/app.py`) loads all six model artifacts once when the server starts, then keeps them in memory for the lifetime of the process. Each `POST /predict` request runs through `inference.py`:

1. **Text cleaning** — the description is lowercased, punctuation is stripped, and NLTK stopwords are removed. Words shorter than 3 characters are also dropped to reduce noise.
2. **POS tagging** — NLTK's `averaged_perceptron_tagger` tags each word with its part of speech. Noun tags (NN, NNS, NNP, NNPS), verb tags (VB, VBD, VBG, VBN, VBP, VBZ), and adjective tags (JJ, JJR, JJS) are counted separately.
3. **Sub-area encoding** — the `sub_area` field is looked up in `sub_area_price_map`. If the locality is not in the training vocabulary (e.g. a newly developed area), the global mean price is used as a fallback.
4. **Amenity encoding** — the amenity score (sum of the 7 binary flags) is looked up in `amenities_score_price_map`. Same fallback applies.
5. **Vectorisation** — `count_vectorizer.transform()` converts the cleaned description into a dense array. Unknown words are silently ignored (the vocabulary is fixed at training time).
6. **Feature assembly** — the 15 structural features and the vectorizer output are concatenated into a single row DataFrame with named columns (from `all_feature_names.pkl`). Named columns are important because the VotingRegressor's sub-estimators were fitted on a named DataFrame.
7. **Prediction + interval** — `model.predict(feature_df)[0]` returns the point estimate. The interval margin is `z_score × residual_std` from `interval_est.pkl`. The lower bound is clamped to 0 (prices can't be negative).

---

## How Experiment Tracking Works

MLflow tracks experiments separately from the DVC pipeline. The two tools have complementary roles:

| Concern | DVC | MLflow |
|---------|-----|--------|
| Are the artifacts reproducible? | Yes — content-hashed |  |
| What hyperparameters were used? | `params.yaml` (current run only) | Every run, queryable |
| How did metrics change across runs? | `dvc metrics diff` (Git-based) | MLflow UI / `mlflow_query.py` |
| Is the model registered for deployment? | | MLflow Model Registry |
| Can I visualise runs side-by-side? | | Yes (parallel coords, scatter) |

When you run `python -m mlops.mlflow_train`, MLflow creates a run under the `M2_Pune_Real_Estate_Price` experiment and logs:

- **Parameters:** all values from `params.yaml`
- **Metrics:** RMSE, MAE, R², interval margin
- **Artifacts:** the trained model (with input/output signature), a residuals-vs-predicted plot, a predicted-vs-actual plot
- **Tags:** `module=M2_Lab5`, `candidate_for_registry=true`

The model signature means MLflow knows the exact input schema — you can later serve it with `mlflow models serve` without writing any serving code.

---

## Quick Start

### Prerequisites

- Python 3.9+
- Git

### 1. Clone & Install

```bash
git clone https://github.com/PRINCENAKOBA/mlops-pune-price-prediction.git
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
| http://localhost:8000/docs | Interactive Swagger UI — try the API in your browser |
| http://localhost:8000/health | Liveness check |
| http://localhost:8000/model/info | Model metadata (type, vocab size, interval margin) |

### 3. Open the Browser UI

Open `frontend/index.html` directly in your browser. Fill in the property details and click **Predict** — the form calls the FastAPI service and displays the predicted price with confidence bounds on the results page.

### 4. Test the API Programmatically

```bash
python src/test_api.py
```

---

## DVC Pipeline

Three stages are defined in `dvc.yaml`. Edit a value in `params.yaml` and `dvc repro` re-runs only the affected downstream stages — unchanged stages are skipped.

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

| What you change | Stages that re-run |
|----------------|--------------------|
| `Pune Real Estate Data.xlsx` | clean → features → train |
| `mlops/clean_data.py` | clean → features → train |
| `mlops/build_features.py` | features → train |
| `params.yaml` (data / ridge / lasso / interval) | train only |
| `mlops/train.py` or `mlops/utils.py` | train only |
| `frontend/*`, `src/*` | none — not part of the DAG |

### Comparing Metrics Across Commits

```bash
dvc metrics show          # metrics at HEAD
dvc metrics diff          # HEAD vs previous commit
dvc metrics diff v1.0     # HEAD vs a tag
```

---

## MLflow Experiment Tracking

All runs are logged to the `M2_Pune_Real_Estate_Price` experiment in a local SQLite database (`mlflow.db`). Switch to DagsHub for shared, hosted tracking (see below).

### Single Training Run

```bash
python -m mlops.mlflow_train
```

Logs all hyperparameters, RMSE / MAE / R², interval margin, diagnostic plots, and model signature.

### Hyperparameter Sweep

```bash
python -m mlops.mlflow_sweep
```

One MLflow run per Ridge alpha in `params.yaml sweep.alphas` (default: `[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]`).

### View Results

```bash
# Ranked leaderboard in the terminal
python -m mlops.mlflow_query

# Full MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

### AutoML Benchmark

```bash
python -m mlops.pycaret_benchmark
```

Benchmarks 20+ algorithms via PyCaret. Results saved to `metrics/pycaret_benchmark.json`.

---

## FastAPI Service

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status": "ok"}` |
| `GET` | `/model/info` | Model type, vectorizer vocab size, interval margin |
| `POST` | `/predict` | Predict property price with confidence interval |

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
| `property_type` | int | Number of bedrooms |
| `area` | float | Total area in square feet |
| `sub_area` | str | Pune locality, e.g. `"kothrud"`, `"baner"`, `"wakad"` |
| `description` | str | Free-text property listing (optional; leave `""` if none) |
| `clubhouse` … `gym` | int | Amenity flags — `1` if present, `0` if absent |

### Example Response

```json
{
  "predicted_price": 85.42,
  "lower_bound": 54.07,
  "upper_bound": 116.77,
  "features_used": 312
}
```

All prices are in **Indian Rupees (₹ lakhs)**. `features_used` is the total number of features passed to the model (15 structural + vocabulary size).

---

## DagsHub Integration

[DagsHub](https://dagshub.com) is a GitHub-like platform for ML projects. It hosts MLflow runs and DVC remote storage, so a team can share experiments and artifacts without running any infrastructure.

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
# Verify credentials are set
python -m mlops.dagshub_setup --check

# Point MLflow at DagsHub for the current process
python -m mlops.dagshub_setup --configure

# Print the dvc remote add commands to push artifacts to DagsHub
python -m mlops.dagshub_setup --print-dvc-cmds
```

Once configured, `mlflow_train` and `mlflow_sweep` log directly to DagsHub. No local MLflow UI needed — view runs at `https://dagshub.com/{user}/{repo}`.

---

## Configuration

All tunable parameters live in `params.yaml`. Code reads them via `mlops.utils.load_params()` — nothing is hard-coded. Change a value and run `dvc repro`; DVC detects the change and re-runs only the affected stages.

```yaml
data:
  test_size: 0.2          # Fraction of rows held out for evaluation
  random_state: 42        # Fixed seed — keep this constant for comparable results

ridge:
  alpha: 1.0              # L2 regularisation strength (higher = more shrinkage)

lasso:
  alpha: 0.1              # L1 regularisation strength
  max_iter: 10000         # Extra iterations needed because the feature matrix is wide

voting:
  weights: null           # null = equal weight for LR / Ridge / Lasso

interval:
  confidence: 0.95        # Coverage for the prediction interval (0.95 = 95%)

sweep:
  alphas: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]   # Ridge alphas to sweep

mlflow:
  experiment_name: M2_Pune_Real_Estate_Price
```

---

## Reproducibility

| Artifact | Versioned by |
|----------|-------------|
| Source code | Git |
| Hyperparameters | `params.yaml` in Git + MLflow run params |
| Intermediate data (`data_cleaned.csv`, `model_features.csv`) | DVC content hash |
| Encoder artifacts (`model/*.pkl`) | DVC content hash |
| Trained model (`model/*.sav`) | DVC pipeline output |
| Experiment metadata | MLflow `run_id` |
| Pipeline definition | `dvc.yaml` in Git |

Full reproduction from scratch:

```bash
git clone https://github.com/PRINCENAKOBA/mlops-pune-price-prediction.git
cd mlops-pune-price-prediction
pip install -r requirements.txt -r requirements-mlops.txt
dvc pull      # fetch data + model artifacts from the DVC remote
dvc repro     # re-run the pipeline → produces identical numbers
```

> **Note:** The raw Pune Real Estate dataset (`Pune Real Estate Data.xlsx`) is proprietary and not redistributed. You need your own copy to run the `clean` stage from scratch. The pre-built model artifacts in `model/` allow you to run the FastAPI service without the raw data.

---

## Tech Stack

| Layer | Tool | Why it was chosen |
|-------|------|--------------------|
| Data pipeline | [DVC](https://dvc.org) | Git-native artifact tracking and stage caching; no separate server needed |
| Experiment tracking | [MLflow 3.x](https://mlflow.org) | Standard, self-hostable; integrates with DagsHub and has a built-in model registry |
| AutoML | [PyCaret 3.x](https://pycaret.org) | Single-call model comparison across 20+ algorithms |
| Remote collaboration | [DagsHub](https://dagshub.com) | Hosted MLflow + DVC remote in one platform; free tier is generous |
| ML models | [scikit-learn](https://scikit-learn.org) | Mature, well-documented; `VotingRegressor` is built-in |
| NLP | [NLTK](https://nltk.org) | Lightweight; POS tagger and stopword lists are all that's needed |
| API | [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) | Async, auto-generates Swagger docs, Pydantic validation built-in |
| Schemas | [Pydantic v2](https://docs.pydantic.dev) | Request validation with clear error messages; tight FastAPI integration |
| Data | [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) | Industry standard for tabular data manipulation |
| Config | `params.yaml` + [PyYAML](https://pyyaml.org) | Version-controlled, human-readable, DVC-aware |
| Frontend | HTML / CSS / JavaScript | No build step, no framework dependency — just open the file |

---

## License

For training and educational use as part of the MLOps course (Module 2). The Pune real estate dataset is proprietary and not included in this repository.
