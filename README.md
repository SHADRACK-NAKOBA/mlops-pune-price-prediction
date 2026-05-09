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
- [Contributors](#contributors)

---

## Project Scope

### What this project is

This is a **production-style MLOps project** built on a real-world dataset of ~200 residential property listings in Pune, India. The goal is to predict the listed price of a property (in Rs lakhs) given its size, location, amenities, and a text description.

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
       |
       v
+--------------+     +-------------------+     +-------------------------+
|  Stage 1     |     |  Stage 2          |     |  Stage 3                |
|  clean       |---->|  features         |---->|  train                  |
|              |     |                   |     |                         |
|  Validate &  |     |  NLP engineering: |     |  VotingRegressor        |
|  transform   |     |  POS counts,      |     |  (Ridge + Lasso + LR)   |
|  raw Excel   |     |  bigrams,         |     |  + interval estimator   |
|              |     |  target encoding  |     |  + DVC metrics JSON     |
+--------------+     +-------------------+     +------------+------------+
                                                            |
                                              +-------------+-------------+
                                              v             v             v
                                         model/*.sav   metrics/      model/*.pkl
                                         (trained     train_metrics  (encoders &
                                          model)       .json          vectorizer)
                                              |
                                              v
                                    +------------------+
                                    |  FastAPI          |
                                    |  src/app.py       |
                                    |                   |
                                    |  POST /predict    |
                                    |  GET  /health     |
                                    |  GET  /model/info |
                                    +--------+----------+
                                             |
                                             v
                                    frontend/index.html
                                    (Browser UI)
```

**DVC** orchestrates the three training stages and caches every artifact — if nothing changed, it skips the stage. **MLflow** sits alongside the pipeline for experiment tracking; it records what each training run produced without being part of the DVC DAG itself. The **FastAPI service** loads the trained artifacts at startup and serves predictions; it has no coupling to DVC or MLflow at runtime.

---

## Results

These are real numbers produced by running the full pipeline on the actual dataset.

### VotingRegressor (Hand-tuned) — `python -m mlops.mlflow_train`

Evaluated on a held-out test set (40 samples, 20% split, `random_state=42`):

| Metric | Value |
|--------|-------|
| R2 | **0.8519** |
| RMSE | **Rs 15.75 lakhs** |
| MAE | **Rs 11.46 lakhs** |
| 95% Prediction Interval Margin | **+/- Rs 31.35 lakhs** |
| Train rows | 159 |
| Test rows | 40 |
| Ridge alpha | 1.0 |
| Lasso alpha | 0.1 |

An R2 of 0.8519 means the model explains 85.2% of the variance in property prices. The RMSE of Rs 15.75 lakhs means predictions are off by roughly Rs 15-16 lakhs on average — reasonable for a 200-row dataset with no external data sources. All parameters and metrics from this run are logged to MLflow automatically for comparison against future runs.

### PyCaret AutoML Benchmark — `python -m mlops.pycaret_benchmark`

PyCaret trained and cross-validated 20+ algorithms on the same train/test split, then tuned and finalized the best one. This ran successfully and produced the following head-to-head comparison:

| Model | RMSE (Rs lakhs) | R2 |
|-------|-----------------|-----|
| Hand-tuned VotingRegressor | 15.75 | 0.8519 |
| **PyCaret Best Pipeline** | **11.79** | **0.9171** |

**What this means:** PyCaret's automated pipeline beats the manual ensemble by delta-R2 = +0.065 (6.5 percentage points) and reduces prediction error by Rs 3.96 lakhs per property. This is expected — AutoML searches a much wider algorithm space (gradient boosting, random forests, ensembles) and applies cross-validated hyperparameter tuning automatically. The hand-tuned linear model remains valuable: it is fully interpretable, fast to retrain, and works well as a production baseline.

**About the LightGBM warnings:** During the benchmark run you will see hundreds of lines like `No further splits with positive gain, best gain: -inf`. These are harmless. They simply mean the 200-row dataset is too small for LightGBM to build trees deeper than 1-2 levels — there is not enough data to find a useful split beyond the root. LightGBM still runs and contributes its result to the leaderboard. PyCaret evaluates every algorithm and picks the winner by cross-validated error regardless of verbose warnings.

### Ridge Alpha Sweep — `python -m mlops.mlflow_sweep`

Logs one MLflow run per alpha value to find the optimal regularisation strength:

| Alpha | R2 |
|-------|----|
| 0.01 | ~0.850 |
| 0.1 | ~0.851 |
| **1.0 (default)** | **0.852** |
| **10.0** | **0.854** |
| 100.0 | ~0.851 |
| 1000.0 | ~0.848 |

Alpha = 10.0 gives a marginal improvement (delta-R2 approx 0.002). The gain is small because the dataset is tiny and linear models are already near their performance ceiling on this feature set. All six runs are visible side-by-side in the MLflow UI.

---

## Project Structure — What Each File Does

```
mlops-pune-price-prediction/
+-- mlops/                   <- All training & MLOps scripts (Lab 5)
+-- src/                     <- FastAPI service (Lab 4)
+-- frontend/                <- Browser UI (Lab 4)
+-- model/                   <- Trained artifacts produced by the pipeline
+-- metrics/                 <- JSON metric files DVC can read and diff
+-- dvc.yaml                 <- Declares the three pipeline stages
+-- params.yaml              <- Single source of truth for hyperparameters
+-- requirements.txt         <- Production dependencies
+-- requirements-mlops.txt   <- MLOps-only dependencies (DVC, MLflow, PyCaret)
```

### `mlops/` — Training & MLOps Scripts

| File | What it does |
|------|-------------|
| `clean_data.py` | **Stage 1 of the DVC pipeline.** Reads the raw Excel file, drops malformed rows, standardises column names, converts price to Rs lakhs, clips outliers, and writes `data_cleaned.csv` (199 rows x 17 columns). |
| `build_features.py` | **Stage 2.** Builds the full feature matrix. Cleans description text (lowercase, strip punctuation, remove stopwords), runs NLTK POS tagging to count nouns/verbs/adjectives, applies target encoding for `sub_area` and `amenity_score`, and fits a CountVectorizer on descriptions. Writes `model_features.csv` (199 x 25 features), `model_target.npy`, and five `.pkl` encoder artifacts that the inference service needs at runtime. |
| `train.py` | **Stage 3.** Loads the feature matrix, splits train/test using params from `params.yaml`, builds a `VotingRegressor` (Ridge + Lasso + LinearRegression with equal weights), fits it, computes an interval estimator from training residuals, and writes the model, interval estimator, and `metrics/train_metrics.json`. This is the DVC entry point — it has no MLflow calls so DVC can run it cleanly. |
| `utils.py` | **Shared utilities** used by every other script. Contains `load_params()` (reads `params.yaml`), `load_features_and_target()`, `train_voting_regressor()`, `score_regressor()`, `compute_interval_estimate()`, and `save_model_artifacts()`. Centralising these means a change to the scoring logic is reflected everywhere automatically. |
| `mlflow_train.py` | Runs the same training as `train.py` but wraps everything in an MLflow run. Logs all params, RMSE/MAE/R2, interval margin, residuals-vs-predicted and predicted-vs-actual diagnostic plots, and the model with an input/output signature. Tags the run as a candidate for the model registry. Results: R2 = 0.8519, RMSE = Rs 15.75L. |
| `mlflow_sweep.py` | Iterates over every alpha in `params.yaml sweep.alphas`, training and logging one MLflow run per value. Only Ridge alpha varies — Lasso and LR stay at their defaults. Useful for understanding how sensitive the model is to regularisation strength. |
| `mlflow_query.py` | Queries all runs in the experiment via the MLflow tracking API, prints a ranked leaderboard sorted by R2, and generates a parallel-coordinates plot showing how each hyperparameter configuration relates to model performance. |
| `pycaret_benchmark.py` | Uses PyCaret's `compare_models()` to evaluate 20+ scikit-learn-compatible algorithms on the same train/test split, tunes the winner, and saves results to `metrics/pycaret_benchmark.json`. Results: PyCaret Pipeline RMSE = Rs 11.79L, R2 = 0.9171 — beats the manual model by 6.5%. |
| `dagshub_setup.py` | A helper that reads three environment variables (`DAGSHUB_USER`, `DAGSHUB_TOKEN`, `DAGSHUB_REPO`) and either verifies them (`--check`), sets the MLflow tracking URI to DagsHub (`--configure`), or prints the exact `dvc remote add` commands (`--print-dvc-cmds`). Never modifies shell config files — only affects the current Python process. |
| `dvc_init.py` | One-time DVC setup: runs `dvc init` and stages the DVC config for the first Git commit. Only needs to be run once per clone. |

### `src/` — FastAPI Inference Service

| File | What it does |
|------|-------------|
| `app.py` | The FastAPI application. Defines three routes (`GET /health`, `GET /model/info`, `POST /predict`), enables CORS for all origins (so the local HTML frontend can call it), and loads all model artifacts once at startup via `inference.py`. |
| `inference.py` | The prediction engine. At startup it loads six artifacts from `model/`: the VotingRegressor, the CountVectorizer, two target-encoding maps (sub-area and amenity score), the feature column names, and the interval estimator. The `predict_price()` function runs a 7-step pipeline (text cleaning → POS tagging → target encoding → vectorisation → feature assembly → prediction → confidence interval) and returns a dict with the point estimate and bounds. |
| `schemas.py` | Pydantic v2 models for the API. `PropertyInput` defines and validates the request body (11 fields). `PriceResponse` defines the response (predicted price, lower bound, upper bound, number of features used). Pydantic rejects bad input automatically with a clear 422 error before it ever reaches the model. |
| `test_api.py` | A simple HTTP client test script. Sends a hard-coded `PropertyInput` to the running server and prints the response. Useful for a quick smoke test after starting the server. Not a unit test suite — just a sanity check. |

### `frontend/` — Browser UI

| File | What it does |
|------|-------------|
| `index.html` | The input form. Has fields for all 11 `PropertyInput` parameters. On submit, `script.js` sends a `POST /predict` request to the FastAPI server and redirects to `results.html`. |
| `results.html` | Displays the prediction result. Shows the point estimate, the lower/upper confidence bounds, and the number of features the model used. |
| `script.js` | The glue between the two HTML pages. Serialises the form into a JSON body, calls `fetch()` against `http://localhost:8000/predict`, and passes the result to the results page via URL query params. |
| `style.css` | Minimal styling. No framework dependencies — just open the file in a browser. |

### `model/` — Trained Artifacts

These files are produced by the DVC pipeline and loaded by the FastAPI service at startup. They are tracked by DVC (not committed to Git directly).

| File | What it contains |
|------|-----------------|
| `property_price_prediction_voting.sav` | The fitted `VotingRegressor`. Serialised with `joblib`. |
| `interval_est.pkl` | A dict with two keys: `z_score` (1.96 for 95% confidence) and `residual_std` (standard deviation of training residuals). At inference time, `margin = z_score x residual_std = +/- Rs 31.35 lakhs`. |
| `count_vectorizer.pkl` | The fitted `CountVectorizer`. Transforms the property description into a bag-of-words matrix. The vocabulary (10 bigrams) is fixed at training time — unknown words at inference are silently ignored. |
| `sub_area_price_map.pkl` | A dict mapping each Pune locality (e.g. `"kothrud"`, `"baner"`) to the mean property price in that area, computed from the training set. This is **target encoding** — it replaces a categorical variable with a numeric signal derived from the label. Unknown localities fall back to the global mean. |
| `amenities_score_price_map.pkl` | Similar target encoding but for amenity count (0-7). Maps the number of amenities a property has to the mean price of training properties with that count. |
| `feature_cols.pkl` | The list of structural feature column names (15 features). Used to build the DataFrame in the correct column order at inference time. |
| `all_feature_names.pkl` | The complete ordered list of all feature names (15 structural + 10 bigram columns = 25 total). Used to create the named DataFrame that the VotingRegressor expects. |

### `metrics/` — DVC Metric Files

| File | What it contains |
|------|-----------------|
| `train_metrics.json` | RMSE (15.75), MAE (11.46), R2 (0.8519), ridge/lasso alphas, train/test counts, interval margin from the most recent `dvc repro` run. Committed to Git so `dvc metrics diff` can compare across commits. |
| `pycaret_benchmark.json` | Head-to-head comparison: PyCaret Pipeline (RMSE 11.79, R2 0.9171) vs Lab 3 VotingRegressor (RMSE 15.75, R2 0.8519). |

### Root Config Files

| File | What it does |
|------|-------------|
| `dvc.yaml` | Defines the three pipeline stages (`clean`, `features`, `train`), their commands, dependencies, parameters, and outputs. DVC reads this to build the DAG and decide which stages to re-run. |
| `params.yaml` | All hyperparameters in one place. DVC watches this file — if a value changes, the affected stage re-runs automatically. Code reads it via `mlops.utils.load_params()`. |
| `requirements.txt` | Dependencies for Labs 1-4: FastAPI, uvicorn, scikit-learn, NLTK, pandas, NumPy, openpyxl, scipy, matplotlib, seaborn. Install this for the API service. |
| `requirements-mlops.txt` | Additional dependencies for Lab 5 only: MLflow, DVC, DagsHub, PyCaret. Kept separate to avoid inflating the API service's dependency footprint. |
| `MLOPS_LAB.md` | Step-by-step guide for completing Lab 5, including exact commands for each tool. |
| `.gitignore` | Excludes virtual environments, Python bytecode, the raw Excel file, DVC cache, MLflow store, NLTK downloads, and OS/IDE artefacts. |
| `.dvcignore` | Tells DVC which files to ignore when computing content hashes (similar to `.gitignore` but for DVC). |

---

## How the ML Pipeline Works

### Stage 1 — Data Cleaning (`mlops/clean_data.py`)

The raw Excel sheet has inconsistent formatting: price columns contain strings like `"85 Lakhs"` or `"1.2 Cr"`, and some rows are missing key fields. `clean_data.py` standardises all of this:

- Loads 200 rows x 18 columns from the Excel file
- Splits `Location` into city / state / country
- Converts all prices to Rs lakhs (crore values multiplied by 100)
- Extracts numeric area and property type via regex
- Encodes 7 amenity columns as 0/1 binary flags
- Drops 1 row with missing price
- Clips outliers at 5th/95th percentiles (area: 422-1663 sqft, price: 36-190 lakhs)
- Writes `data_cleaned.csv` with 199 rows x 17 columns

### Stage 2 — Feature Engineering (`mlops/build_features.py`)

This stage turns the clean tabular data into a numeric feature matrix the model can use. It produces 25 features total.

**Structural features (15 columns)**

| Feature | Type | Description |
|---------|------|-------------|
| `property_type` | int | Number of bedrooms |
| `area` | float | Area in sq ft |
| `clubhouse`, `school`, `hospital`, `mall`, `park`, `pool`, `gym` | int (0/1) | Amenity presence flags |
| `price_by_subarea` | float | Target-encoded locality mean price |
| `amenity_score` | int | Sum of the 7 amenity flags (0-7) |
| `price_by_amenities` | float | Target-encoded amenity-count mean price |
| `noun_count`, `verb_count`, `adj_count` | int | POS tag counts from the description |

**Target encoding** replaces high-cardinality categorical columns (locality, amenity count) with the mean target value for that category, computed on the training split. This lets the model use location and amenity richness as a price signal without one-hot encoding dozens of dummy columns.

**NLP text features (10 bigram columns)**

The property description is cleaned (lowercase, strip punctuation, remove stopwords), then a `CountVectorizer` with bigrams (pairs of consecutive words) and a vocabulary of 10 is fitted. The 10 most informative bigrams found were: `bedroom apartment`, `boasts elegant`, `community living`, `elegant towers`, `mantra gold`, `offering bedroom`, `project boasts`, `project offers`, `stories offering`, `towers stories`. Each becomes a feature column with the count of how many times it appears in the description.

**POS tagging** (Part-of-Speech tagging using NLTK) counts nouns, verbs, and adjectives separately. Listings with many adjectives tend to describe higher-value properties more elaborately.

### Stage 3 — Training (`mlops/train.py`)

The model is a `VotingRegressor` combining three estimators with equal weight:

- **Ridge Regression** — linear regression with L2 regularisation (alpha=1.0). Shrinks all coefficients towards zero, reducing overfitting on text features.
- **Lasso Regression** — linear regression with L1 regularisation (alpha=0.1). Drives some coefficients to exactly zero, performing automatic feature selection.
- **Linear Regression** — ordinary least squares, unregularised. Acts as an anchor to prevent over-regularisation.

The ensemble averages the three predictions. Each estimator has different biases and the average is more stable than any single one.

**Prediction interval estimation** — after fitting, the model predicts on the training set and computes the standard deviation of the residuals (`residual_std`). At inference time, the 95% interval is `prediction +/- 1.96 x residual_std`. This gives a +/- Rs 31.35 lakh band around every prediction.

---

## How the API Works

The FastAPI service (`src/app.py`) loads all model artifacts once at startup and keeps them in memory. Each `POST /predict` request runs through `inference.py` in 7 steps:

1. **Text cleaning** — the description is lowercased, punctuation is stripped, NLTK stopwords are removed, and words shorter than 3 characters are dropped.
2. **POS tagging** — NLTK's `averaged_perceptron_tagger` tags each word. Noun, verb, and adjective counts are extracted separately.
3. **Sub-area encoding** — the `sub_area` field is looked up in `sub_area_price_map`. If the locality is not in the training vocabulary, the global mean price is used as a fallback.
4. **Amenity encoding** — the amenity score (sum of 7 binary flags) is looked up in `amenities_score_price_map`. Same fallback applies.
5. **Vectorisation** — `count_vectorizer.transform()` converts the cleaned description into bigram counts. Unknown words are silently ignored.
6. **Feature assembly** — 15 structural features and 10 bigram features are concatenated into a named DataFrame. Named columns are required because the VotingRegressor's sub-estimators were fitted on a named DataFrame.
7. **Prediction + interval** — `model.predict(feature_df)[0]` returns the point estimate. The interval margin is `1.96 x residual_std`. The lower bound is clamped to 0 (prices cannot be negative).

---

## How Experiment Tracking Works

MLflow tracks experiments separately from the DVC pipeline. The two tools have complementary roles:

| Concern | DVC | MLflow |
|---------|-----|--------|
| Are the artifacts reproducible? | Yes — content-hashed | |
| What hyperparameters were used? | `params.yaml` (current run only) | Every run, queryable |
| How did metrics change across runs? | `dvc metrics diff` (Git-based) | MLflow UI / `mlflow_query.py` |
| Is the model registered for deployment? | | MLflow Model Registry |
| Can I visualise runs side-by-side? | | Yes (parallel coords, scatter) |

**What `python -m mlops.mlflow_train` actually does:**

1. Sets the tracking URI to `sqlite:///mlflow.db` (a local database file) unless `MLFLOW_TRACKING_URI` is already set in the environment
2. Creates or reuses the `M2_Pune_Real_Estate_Price` experiment
3. Starts a new MLflow run named `voting_regressor`
4. Trains the VotingRegressor on the training split
5. Logs all hyperparameters (ridge_alpha=1.0, lasso_alpha=0.1, test_size=0.2, n_features=25, etc.)
6. Logs metrics: RMSE=15.75, MAE=11.46, R2=0.8519, interval_margin=31.35
7. Creates and logs two diagnostic plots as artifacts: residuals-vs-predicted and predicted-vs-actual
8. Logs the trained model with its input/output signature (so `mlflow models serve` can deploy it later)
9. Saves the same model to `model/` so the FastAPI service can pick it up immediately

**What `python -m mlops.pycaret_benchmark` actually does:**

1. Loads the same feature matrix used by the VotingRegressor
2. Runs PyCaret `setup()` — initialises preprocessing (normalisation, train/val split)
3. Runs `compare_models(n_select=10, sort="R2")` — trains ~20 algorithms with 5-fold cross-validation each and ranks them by R2
4. Tunes the top algorithm with `tune_model(n_iter=20)` — searches 20 hyperparameter combinations
5. Finalises the model by refitting on the full training pool
6. Scores it on the Lab 3 test set and prints a head-to-head table
7. Saves the PyCaret pipeline (preprocessing + model) to `model/pycaret_pune_real_estate.pkl`
8. Saves the comparison to `metrics/pycaret_benchmark.json`

To view all logged MLflow runs in the UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- Git

### 1. Clone & Install

```bash
git clone https://github.com/SHADRACK-NAKOBA/mlops-pune-price-prediction.git
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

# Place Pune Real Estate Data.xlsx in the project root, then:
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

Logs all hyperparameters, RMSE / MAE / R2, interval margin, diagnostic plots, and model signature. Produces run ID you can use to retrieve the model later.

### Hyperparameter Sweep

```bash
python -m mlops.mlflow_sweep
```

One MLflow run per Ridge alpha in `params.yaml sweep.alphas` (default: `[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]`). View all runs side-by-side in the UI.

### Query and Compare Runs

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

Benchmarks 20+ algorithms via PyCaret. Results saved to `metrics/pycaret_benchmark.json`. Expect LightGBM warnings about no positive splits — these are harmless on a 200-row dataset.

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
| `clubhouse` to `gym` | int | Amenity flags — `1` if present, `0` if absent |

### Example Response

```json
{
  "predicted_price": 85.42,
  "lower_bound": 54.07,
  "upper_bound": 116.77,
  "features_used": 25
}
```

All prices are in **Indian Rupees (Rs lakhs)**. `features_used` is the total number of features passed to the model (15 structural + 10 bigram text features = 25).

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
  random_state: 42        # Fixed seed — keep constant for comparable results

ridge:
  alpha: 1.0              # L2 regularisation strength (higher = more shrinkage)

lasso:
  alpha: 0.1              # L1 regularisation strength
  max_iter: 10000         # Extra iterations needed for the wide feature matrix

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
git clone https://github.com/SHADRACK-NAKOBA/mlops-pune-price-prediction.git
cd mlops-pune-price-prediction
pip install -r requirements.txt -r requirements-mlops.txt
# Place Pune Real Estate Data.xlsx in the project root
dvc repro     # runs all three stages → identical numbers
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
| NLP | [NLTK](https://nltk.org) | Lightweight; POS tagger and stopword lists are all that is needed |
| API | [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) | Async, auto-generates Swagger docs, Pydantic validation built-in |
| Schemas | [Pydantic v2](https://docs.pydantic.dev) | Request validation with clear error messages; tight FastAPI integration |
| Data | [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org) | Industry standard for tabular data manipulation |
| Config | `params.yaml` + [PyYAML](https://pyyaml.org) | Version-controlled, human-readable, DVC-aware |
| Frontend | HTML / CSS / JavaScript | No build step, no framework dependency — just open the file |

---

## Contributors

**SHADRACK NAKOBA**

---

## License

For training and educational use as part of the MLOps course (Module 2). The Pune real estate dataset is proprietary and not included in this repository.
