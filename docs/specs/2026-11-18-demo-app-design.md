# Team 8 – Bike-Sharing Demand Demo App: Design

Date: 2026-11-18 · Status: approved by the team (2026-11-18)

## 1. Intent

The combined PBL 2 + PBL 3 review needs more than a notebook. The team wants a working
app it can open in front of the faculty and say: "this is the model we built, running on
real inputs." The app must show work from every unit (mainly 3, 4 and 5) using the models
the notebooks already trained and verified. It is a demo of the ML work, not a product.

**Success:** during the viva, any team member opens a public link (or runs it locally if
there is no internet), changes inputs, and gets predictions, a 24-hour forecast and the
cluster/state views within a second or two, with numbers that match the report.

## 2. Goals, non-goals, anti-goals

Goals
- Live prediction from user inputs (Unit 5 classifier + GAM count).
- Live 24-hour Kalman forecast from any point in the test period (Unit 4).
- Visual pages for clustering, PCA and HMM states (Units 3–4).
- One leaderboard page with the metrics of all five units.
- Free public deployment and a local fallback.

Non-goals
- No user accounts, database, or live data feed.
- No retraining inside the app.
- No new ML methods beyond what the notebooks contain.

Anti-goals (failure even if it "works")
- Any number in the app that differs from the notebook or report.
- A demo that depends on a slow cold-start server or breaks on stage.
- Hidden leakage: the app must not use `casual` / `registered`, and must use the same
  chronological 80/20 split and the threshold of 130 as the notebooks.

## 3. Constraints

- Data: UCI Bike Sharing (id 275), `hour.csv` / `day.csv`, years 2011–2012 only.
- Reference notebooks: `Team8_Bike_Sharing_ML_PBL1_Final.ipynb` (Units 1–2) and
  `Team8_Bike_Sharing_ML_PBL2_Units3-5.ipynb` (Units 3–5). Reference numbers:
  `results.json` and `results_units345.json`.
- Local environment: Python 3.13, numpy 2.5, pandas 3.0, scikit-learn 1.9, scipy 1.18.
  Streamlit is not installed yet.
- Pickled scikit-learn models only load reliably under the same scikit-learn version, so
  versions are pinned.

## 4. Approaches considered

| | Approach | Trade-off | Verdict |
|---|---|---|---|
| A | **Streamlit multipage app**, models loaded in-process, deployed on Streamlit Community Cloud | One Python codebase that reuses the notebook code directly. Free hosting, 1–2 days of work. It looks like a dashboard, not a "full-stack" product | **Chosen by the team** |
| B | FastAPI backend + React frontend (Render + Vercel) | Looks more full-stack, but 3–4× the work, two deploys, and the free backend can take a long time to wake up during the viva | Rejected |
| C | Gradio on Hugging Face Spaces | Quickest, but looks generic and is weak for multi-chart dashboards | Rejected |

## 5. Design

### 5.1 Repository layout (`D:\ML Project SRMIST`)

```
data/            hour.csv, day.csv, ML_Project_Batch_details.xlsx
notebooks/       PBL1 and PBL2/3 notebooks (unchanged copies)
pbl1_archive/    previous PDFs, PPTX, audit notes (read-only reference)
training/
  train_models.py   rebuilds the demo models + precomputed views, writes artifacts/
artifacts/       *.joblib models, *.json params/metrics, *.csv precomputed views
app/
  Home.py                       landing page: problem, team, how to use
  pages/1_Predict.py
  pages/2_Forecast_24h.py
  pages/3_Demand_Patterns.py
  pages/4_Model_Leaderboard.py
  core.py                       shared feature building + artifact loading (cached)
tests/test_app.py               smoke tests
requirements.txt                pinned versions
docs/specs/                     this file
report/ slides/                 later deliverables
```

### 5.2 Training/export script (`training/train_models.py`)

- Loads `data/hour.csv` and applies the notebook's split (first 80 % of rows = train).
- Re-fits only what the app needs, with the notebook's exact settings:
  - Unit 5 stacking classifier: CART + RF + GB + AdaBoost base models, logistic meta-model
    fitted on expanding-window out-of-fold predictions.
  - Unit 5 Poisson spline GAM (count model), with the alpha chosen in the notebook.
  - Unit 4 Kalman parameters (φ, Q, R) and the 48-cell hour × workingday profile.
- Writes precomputed views: day-level K-means labels and profiles (K = 2), PCA 2-D
  coordinates, and HMM Viterbi states for the test period.
- **Self-check:** recomputes the test metrics and asserts they match
  `results_units345.json` (e.g. stacking AUC 0.978, GAM MAE 44.9, Kalman 1-h MAE 31.2)
  to the printed precision. The script fails loudly if they drift.
- Saves models with `joblib` (compressed). Run once; the artifacts are committed.

### 5.3 Pages

**Home** – one-paragraph problem statement, team names and register numbers, a pipeline
diagram (data → preprocessing → Units 1–5 → app), and a guide to the pages.

**1 · Predict** (Unit 5)
```
Date [2012-08-15]  Hour [08]  Weather [Clear ▾]
Temp °C [24]  Humidity % [60]  Wind km/h [12]
──────────────────────────────────────────────
Predicted rentals this hour:   ~ 512   (GAM)
High-demand probability:       0.94 → HIGH (> 130 rides)   (Stacking)
[line] predicted rentals vs temperature, other inputs fixed (GAM effect)
[bar ] base-model probabilities: CART / RF / GB / Ada / Stack
```
- Season, month, weekday, holiday and workingday are derived from the date. For
  2011–2012 the dataset's own holiday flag is used; for other dates the US federal
  holiday calendar is used.
- Dates outside 2011–2012 are mapped to `yr = 2012`, with a visible note that the model
  only knows two years of growth.
- Raw inputs are converted to the dataset's normalised scales (temp/41, hum/100,
  windspeed/67). If the model needs `atemp`, it is estimated from temp with a linear fit
  on the training rows (builder's call, documented in the code).

**2 · Forecast 24h** (Unit 4)
- Pick any date and hour in the test period (7 Aug – 31 Dec 2012).
- Run the Kalman filter on all hours up to that point, then forecast the next 24 hours:
  profile + level, back-transformed with `expm1`, with a 95 % band.
- Plot the forecast against the actual counts and two baselines (persistence, same hour
  yesterday), with the MAE of each over that 24-hour window.

**3 · Demand Patterns** (Units 3–4)
- Tabs: K-means day profiles (commute vs leisure shape) · PCA 2-D map coloured by cluster
  or workingday · HMM decoded states over a selected test week, drawn over the counts.
- Figures are drawn from the precomputed CSVs with Altair (bundled with Streamlit, no
  extra dependency).

**4 · Model Leaderboard** (Units 1–5)
- Tables built from `results.json` and `results_units345.json`: regression (OLS, ridge,
  Bayesian, GAM, GB), classification (logistic, SVM, CART, RF, GB, Ada, stacking),
  forecasting (Kalman vs HMM vs baselines), clustering (K-selection table).
- A short "what this shows" line under each table, consistent with the report.

### 5.4 Shared code (`app/core.py`)

- `load_artifacts()` is cached with `st.cache_resource`, so models load once per session.
- `build_features(inputs) -> DataFrame` gives one code path for the GAM and classifier
  inputs, and mirrors the notebook's column construction exactly.
- `kalman_forecast(history, phi, Q, R, profile, horizon=24)` is copied from the
  notebook's `kalman` function.

### 5.5 Deployment

- GitHub repository (created from the team's own account) → Streamlit Community Cloud,
  Python 3.13, `requirements.txt` pinned to the versions used for training
  (`scikit-learn==1.9.0` etc.).
- Local fallback for the viva: `pip install -r requirements.txt` then
  `streamlit run app/Home.py`.
- The dataset is CC BY 4.0, so publishing it in the repository is allowed (with
  attribution in the README).

### 5.6 Testing

- `train_models.py` self-check (metric parity with the notebook).
- `tests/test_app.py` uses `streamlit.testing.v1.AppTest` to run every page with default
  inputs and assert there is no exception. It also checks a known test-set row gives the
  same stacking probability as the notebook.
- A manual click-through in the browser before the review.

## 6. Decided by the team
- Approach A (Streamlit).

## 7. Builder's calls (marked as such, open to correction)
- The four pages above; the "what-if" view is merged into Predict.
- Altair for charts; no Plotly.
- The atemp estimate and the holiday rule for out-of-range dates.
- Artifacts are committed to git (expected size is well under 100 MB after compression).

## 8. Still open (not blocking the app build)
- Review date (sets how much polish fits).
- Whether faculty accept the same hourly data for Unit 4 or want a separate sequential
  dataset. If a new dataset is required, the Forecast page gains a second dataset.
- The GitHub account to deploy from.
- Colab links, PBL 1 feedback, team roles: needed for the report, not the app.

## 9. Later deliverables (separate designs, after the app)
1. Integrated report in the SRMIST template (DOCX + PDF).
2. Combined PBL 2+3 slide deck with demo and architecture slides, demo script, viva notes.
