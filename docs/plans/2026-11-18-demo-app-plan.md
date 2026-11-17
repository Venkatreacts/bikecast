# Bike-Sharing Demand Demo App – Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A 4-page Streamlit app that serves Team 8's verified Unit 3–5 models (stacking classifier, Poisson GAM, Kalman filter, K-means/PCA/HMM views) with numbers identical to the notebook.

**Architecture:** `training/train_models.py` executes the existing Units 3–5 notebook's code cells unchanged in one namespace, asserts its results equal the reference `results_units345.json`, then exports the fitted objects and precomputed views to `artifacts/`. The app (`app/`) only loads artifacts and the CSV data; `app/core.py` holds every computation so pages stay thin and testable.

**Tech Stack:** Python 3.13, numpy 2.5.2, pandas 3.0.1, scikit-learn 1.9.0, scipy 1.18.0, joblib 1.5.3, matplotlib 3.10.8, seaborn (training only), Streamlit (+ bundled Altair), pytest.

**Spec:** `docs/specs/2026-11-18-demo-app-design.md`

## Global Constraints

- Repo root: `D:\ML Project SRMIST`. All paths below are relative to it.
- Split: first `int(0.8 * 17379) = 13903` rows of `hour.csv` sorted by `(date, hr)` = train; rest = test. Threshold `THRESH = 130.0` (training median); high demand = `cnt > 130`.
- Never use `casual` or `registered` as inputs.
- Classifier feature order (`RAW_COLS`): `['season','yr','mnth','hr','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed']`.
- Normalisation (UCI): `temp = °C/41`, `atemp = °C/50`, `hum = %/100`, `windspeed = km/h/67`.
- Pinned versions in `requirements.txt` must equal the versions used when `train_models.py` produced the committed artifacts.
- Charts: Altair only (no Plotly). Every chart has a title and a one-line caption.
- Numbers shown in the app come from artifacts or reference JSON, never typed by hand.

## Review Focus

1. **Extreme inputs on Predict** (temp −10 °C or 45 °C, humidity 0 or 100 %, wind 0): predictions must be finite, count ≥ 0, probability in [0, 1]. → test in Task 3.
2. **Forecast origin near the end of the data** (e.g. 31 Dec 2012 23:00, fewer than 24 actual hours left): forecast still drawn for 24 steps; actuals and MAE use only the hours that exist; no exception. → test in Task 5.
3. **Calendar edge cases**: a federal holiday on a weekday → `workingday = 0`; a date outside 2011–2012 → `yr = 1` with an `out_of_range` flag shown to the user. → test in Task 3.
4. **Missing clock-hours** (165 hours absent in `hour.csv`): the Kalman filter runs over rows in order, exactly as the notebook does; forecast steps are labelled `+1 h … +24 h` from the chosen row, not by a reconstructed clock. → test in Task 3 (parity with the notebook proves the row convention).
5. **scikit-learn version mismatch at load time** (e.g. Streamlit Cloud installs another version): the app shows a clear error naming both versions instead of crashing on unpickling. → test in Task 3.

---

### Task 1: Repository setup

**Files:**
- Create: `data/hour.csv`, `data/day.csv`, `data/ML_Project_Batch_details.xlsx` (copied)
- Create: `notebooks/Team8_Bike_Sharing_ML_PBL1_Final.ipynb`, `notebooks/Team8_Bike_Sharing_ML_PBL2_Units3-5.ipynb` (copied, unchanged)
- Create: `reference/results.json`, `reference/results_units345.json` (copied, unchanged)
- Create: `pbl1_archive/` (copy every other file from `D:\Users\Admin\Downloads\Team8_ML_PBL1_Deliverables\Team8_ML_PBL1_Deliverables\`, including `figures/`; skip `__MACOSX`)
- Create: `requirements.txt`, `.gitignore`, `tests/test_layout.py`

**Interfaces:**
- Produces: the directory layout above; `build/` is git-ignored scratch space.

- [ ] **Step 1: Copy the files** listed above from the source folder. Do not modify them.
- [ ] **Step 2: `git init`**, then add `.gitignore` containing `build/`, `__pycache__/`, `.pytest_cache/`, `*.pyc`, `.streamlit/secrets.toml`.
- [ ] **Step 3: Install** `pip install streamlit seaborn` and record the installed versions with `pip show`.
- [ ] **Step 4: Write `requirements.txt`** pinning `numpy==2.5.2`, `pandas==3.0.1`, `scikit-learn==1.9.0`, `scipy==1.18.0`, `joblib==1.5.3`, `matplotlib==3.10.8`, and the exact `seaborn` and `streamlit` versions from Step 3. Also add `pytest` (unpinned).
- [ ] **Step 5: Write the test** `tests/test_layout.py::test_data_present`:
```python
hour = pd.read_csv("data/hour.csv"); day = pd.read_csv("data/day.csv")
assert hour.shape == (17379, 17) and day.shape == (731, 16)
assert json.load(open("reference/results_units345.json"))["kalman"]["phi"] == pytest.approx(0.93493, abs=1e-5)
```
- [ ] **Step 6: Run** `pytest tests/test_layout.py -v` → PASS.
- [ ] **Step 7: Commit** `chore: project layout, data, reference results`.

---

### Task 2: Training and export script

**Files:**
- Create: `training/train_models.py`
- Test: `tests/test_training.py`

**Interfaces:**
- Produces:
  - `run_notebook(nb_path: Path, workdir: Path) -> dict` executes every code cell of the notebook in one namespace, with `cwd = workdir`, `matplotlib.use("Agg")`, `display` stubbed to a no-op and `plt.show` to `plt.close("all")`. Returns the namespace.
  - `check_parity(R: dict, ref: dict) -> list[str]` returns one message per mismatch, comparing with `abs=1e-3`: `ref["kalman"]["phi"]`, `["Q"]`, `["R"]`; the `MAE` of the row `Model == "Kalman LDS, 1 h ahead"` in `["forecast"]`; `ROC-AUC` and `Accuracy` of the row whose `Model` starts with `"Stacking"` in `["cls"]`; `MAE` of `"Poisson GAM (splines)"` in `["gam"]["table"]`; `["kmeans"]["sil"]`.
  - `export(ns: dict, out: Path) -> None` writes:
    - `artifacts/models.joblib` (compress=3): `{"base": ns["fitted_base"], "meta": ns["meta"], "gam": ns["gam"], "thresh": float(ns["THRESH"]), "atemp_fit": np.polyfit(train.temp, train.atemp, 1).tolist(), "sklearn_version": sklearn.__version__}`
    - `artifacts/kalman.json`: `{"phi","Q","R"}` from `phi_h, Q_h, R_h`, `"profile": m_tr.tolist()` (48 values, index `hr*2 + workingday`), `"P0": 1.0`
    - `artifacts/kmeans_profiles.csv`: columns `cluster, hr, share` from `prof_km`
    - `artifacts/days.csv`: one row per clustered day (`M.index`): `date, cluster (lab_km), workingday, total, pc1, pc2` (`pc1, pc2` = first two columns of `Zp`)
    - `artifacts/hmm_test.csv`: one row per test hour: `date, hr, cnt, state (path_te), state_cnt = expm1(hm.mu[state])`
    - `artifacts/parity.json`: `{"test_idx": [0,1,2,3,4,100,2000], "p_stack": p[those], "gam": pg[those], "kalman": [{"t0": t, "z1": z1[t], "var1": Pp_all[t] + R_h, "z24": z24[t+23], "var24": v24[t+23]} for t in (n_train+100, n_train+2000)]}` (test indices are offsets into `test`)
  - CLI: `python training/train_models.py` copies `data/*.csv` into `build/`, runs the notebook there, prints mismatches and exits 1 if any, otherwise exports and prints `OK`.

- [ ] **Step 1: Write the failing tests** in `tests/test_training.py`:
  - `test_check_parity_identical`: `check_parity(ref, ref) == []` with `ref` = reference JSON.
  - `test_check_parity_detects_drift`: deep-copy `ref`, set the stacking row's `ROC-AUC` to `0.90` → exactly one message, containing `"ROC-AUC"`.
- [ ] **Step 2: Run** `pytest tests/test_training.py -v` → FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement** `run_notebook`, `check_parity`, `export`, `main` in `training/train_models.py`.
- [ ] **Step 4: Run** `pytest tests/test_training.py -v` → PASS.
- [ ] **Step 5: Run the full export** `python training/train_models.py` (≈5 min) → prints `OK`. If parity fails, **stop and report the mismatches to the user**; do not loosen tolerances.
- [ ] **Step 6: Add** `tests/test_training.py::test_artifacts_load`: every file above exists; `joblib.load("artifacts/models.joblib")["base"]` has 4 models; `len(json.load(open("artifacts/kalman.json"))["profile"]) == 48`; `days.csv` has 728 rows and 2 clusters.
- [ ] **Step 7: Run** `pytest -v` → PASS. Check `du -sh artifacts` < 90 MB.
- [ ] **Step 8: Commit** `feat: training script with notebook parity check and exported artifacts`.

---

### Task 3: Shared core (`app/core.py`)

**Files:**
- Create: `app/core.py`, `app/__init__.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Consumes: Task 2 artifacts.
- Produces:
  - `load_artifacts() -> dict` with keys `models, kalman, kmeans_profiles, days, hmm_test, hour (DataFrame of data/hour.csv sorted by date,hr), ref (results_units345.json), ref_pbl1 (results.json)`; decorated with `st.cache_resource` in pages, not here. If `models["sklearn_version"] != sklearn.__version__`, raise `ArtifactVersionError(saved, installed)`.
  - `calendar_fields(d: datetime.date, day_df: pd.DataFrame) -> dict` returns `season, yr, mnth, holiday, weekday, workingday, out_of_range`. Rules: holiday from `day.csv` for 2011–2012, else `pandas.tseries.holiday.USFederalHolidayCalendar`; `weekday` 0 = Sunday; `workingday = weekday not in (0,6) and not holiday`; season 2 from 21 Mar, 3 from 21 Jun, 4 from 23 Sep, 1 from 21 Dec; `yr = 0` for 2011 else `1`; `out_of_range = year not in (2011, 2012)`.
  - `make_row(d, hr: int, weathersit: int, temp_c: float, hum_pct: float, wind_kmh: float, day_df, atemp_fit) -> pd.DataFrame` returns one row with all `RAW_COLS` (`atemp = polyval(atemp_fit, temp)`).
  - `predict(models: dict, rows: pd.DataFrame) -> pd.DataFrame` with columns `count_gam`, `p_cart, p_rf, p_gb, p_ada, p_stack`, `high (bool, p_stack >= .5)`. GAM inputs are built from `rows` exactly like the notebook's `gdat`; stacking uses `meta.predict_proba(logit(clip(base probs, 1e-4, 1-1e-4)))`.
  - `kalman_forecast(cnt: np.ndarray, workingday: np.ndarray, hr: np.ndarray, t0: int, k: dict, horizon: int = 24) -> pd.DataFrame` with columns `step (1..horizon), mean, lo95, hi95` on the count scale. It filters residuals `log1p(cnt) - profile[hr*2+wd]` over rows `0..t0-1` (`l0=0`, `P0=1.0`), then for step h: `z = profile[hr*2+wd of row t0-1+h] + phi**h * l`, `var = phi**(2h)*P + Q*(1-phi**(2h))/(1-phi**2) + R`, `mean = expm1(z)`, bounds `expm1(z ± 1.96*sqrt(var))`. Rows past the end of the data reuse the hour/workingday pattern from one day earlier (shortcut: assumes the next day has the same working-day flag).

- [ ] **Step 1: Write the failing tests** in `tests/test_core.py`:
  - `test_calendar_matches_day_csv`: for all 731 rows of `day.csv`, `calendar_fields` reproduces `season, yr, mnth, holiday, weekday, workingday`.
  - `test_calendar_edges`: `calendar_fields(date(2012,7,4))` → `holiday=1, workingday=0, weekday=3`; `calendar_fields(date(2025,3,10))` → `yr=1, out_of_range=True, workingday=1`.
  - `test_predict_parity`: `predict` on `hour` test rows `parity["test_idx"]` equals `parity["p_stack"]` and `parity["gam"]` to `abs=1e-9`.
  - `test_kalman_parity`: for each `parity["kalman"]` entry, `kalman_forecast(..., t0)` step 1 mean equals `expm1(z1)` and step 24 equals `expm1(z24)` to `rel=1e-9`.
  - `test_extreme_inputs`: for every combination of temp_c ∈ {−10, 45}, hum ∈ {0, 100}, wind ∈ {0, 67}: `count_gam` finite and ≥ 0, `0 ≤ p_stack ≤ 1`.
  - `test_forecast_at_end`: `t0 = len(hour) - 1` returns 24 finite rows.
  - `test_version_mismatch`: monkeypatch `sklearn.__version__ = "0.0"` → `load_artifacts()` raises `ArtifactVersionError` whose message contains both versions.
- [ ] **Step 2: Run** `pytest tests/test_core.py -v` → FAIL.
- [ ] **Step 3: Implement** the functions above in `app/core.py`.
- [ ] **Step 4: Run** `pytest tests/test_core.py -v` → PASS.
- [ ] **Step 5: Commit** `feat: core prediction, calendar and Kalman forecast with notebook parity`.

---

### Task 4: Home and Predict pages

**Files:**
- Create: `app/Home.py`, `app/pages/1_Predict.py`, `app/ui.py`
- Test: `tests/test_pages.py`

**Interfaces:**
- Consumes: `app/core.py`.
- Produces: `ui.get_artifacts()` = `st.cache_resource`-wrapped `load_artifacts()`; it catches `ArtifactVersionError` and calls `st.error(str(e)); st.stop()`. Every page calls it.

- [ ] **Step 1: Write the failing test** `tests/test_pages.py::test_page_runs[Home.py, pages/1_Predict.py]` (parametrised): `AppTest.from_file(path, default_timeout=60).run()`; assert `not at.exception`. For Predict, also assert at least 2 `at.metric` elements.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement Home:** title "Bike-Sharing Demand Forecasting for Smart Urban Mobility", Team 8 names and register numbers (from `data/ML_Project_Batch_details.xlsx` via the spec's list: Ali Hasan Ansari RA2411026011160, Fayas K N RA2411026011161, Konanki Revanth Chowdary RA2411026011162, Venkatanathan O R V RA2411026011163, Dhruva Mishra RA2411026011164), the problem in 3 sentences, a pipeline diagram (`st.graphviz_chart`: data → preprocessing → Units 1–5 → app pages), and one line per page.
- [ ] **Step 4: Implement Predict:** inputs date (default 2012-08-15), hour (0–23, default 8), weather (1 Clear, 2 Mist, 3 Light rain/snow, 4 Heavy rain), temp °C (−10…45, default 24), humidity % (0…100, default 60), wind km/h (0…67, default 12). Show `st.metric` "Predicted rentals (GAM)" and "High-demand probability (Stacking)" with label HIGH/LOW and the 130 threshold; a bar chart of the 5 model probabilities; a line chart of `count_gam` vs temp from −5 to 40 °C with other inputs fixed. If `out_of_range`, `st.info` that the model only knows 2011–2012 and treats the date as 2012.
- [ ] **Step 5: Run** `pytest tests/test_pages.py -v` → PASS.
- [ ] **Step 6: Commit** `feat: home and predict pages`.

---

### Task 5: Forecast 24h page

**Files:**
- Create: `app/pages/2_Forecast_24h.py`
- Test: `tests/test_pages.py` (extend)

- [ ] **Step 1: Extend the test**: add `pages/2_Forecast_24h.py` to the parametrised list; add `test_forecast_last_hour`: set the date input to 2012-12-31 and hour to 23, run, `not at.exception`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement:** date picker limited to the test period (2012-08-07 … 2012-12-31) and hour select; `t0` = index of the first row at or after that date-hour. Call `kalman_forecast`. Show an Altair chart of steps +1…+24: Kalman mean with the 95 % band, actual counts (where rows exist), persistence (`cnt[t0-1]` repeated) and same-hour-yesterday (`cnt[t0-24+h-1]`). Show a table of MAE per method over the available actual hours. Caption: the full-test-period numbers from `ref["forecast"]` (Kalman 1 h MAE 31.2, persistence 85.2, same hour yesterday 80.5).
- [ ] **Step 4: Run** `pytest tests/test_pages.py -v` → PASS.
- [ ] **Step 5: Commit** `feat: 24-hour Kalman forecast page`.

---

### Task 6: Demand Patterns and Model Leaderboard pages

**Files:**
- Create: `app/pages/3_Demand_Patterns.py`, `app/pages/4_Model_Leaderboard.py`
- Test: `tests/test_pages.py` (extend)

- [ ] **Step 1: Extend the test**: add both pages to the parametrised list; for the leaderboard, assert at least 4 `at.dataframe` elements.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement Demand Patterns** with 3 tabs:
  - K-means: line chart of `kmeans_profiles.csv` (share vs hour, one line per cluster) + the K-selection table from `ref["kmeans_table"]`. Caption: silhouette and DB both choose K = 2; ARI with working day 0.94.
  - PCA: scatter of `days.csv` `pc1` vs `pc2`, colour switch between cluster and workingday.
  - HMM: week selector over the test period (default 15–21 Oct 2012); chart of `cnt` with `state_cnt` as a step line, colour by state; the transition matrix from `ref["hmm"]["A"]` as a table.
- [ ] **Step 4: Implement Model Leaderboard** with 4 tables, each with a one-line "what this shows" caption: Unit 2 regression (`ref_pbl1["regression"]`), classification (`ref_pbl1["classification"]` + `ref["cls"]`, sorted by ROC-AUC), Unit 4 forecasting (`ref["forecast"]`), Unit 5 GAM vs regressors (`ref["gam"]["table"]`).
- [ ] **Step 5: Run** `pytest -v` → all PASS.
- [ ] **Step 6: Run locally** `streamlit run app/Home.py`, click through every page in the browser, change each input once, and screenshot each page into `docs/screenshots/` (later used in the report).
- [ ] **Step 7: Commit** `feat: demand patterns and leaderboard pages`.

---

### Task 7: README and deployment

**Files:**
- Create: `README.md`, `.streamlit/config.toml`

- [ ] **Step 1: Write `README.md`**: project title, team, how to run locally (`pip install -r requirements.txt`, `streamlit run app/Home.py`), how to regenerate artifacts (`python training/train_models.py`), the dataset citation (Fanaee-T & Gama, UCI Bike Sharing Dataset, CC BY 4.0), and the live link (filled in at Step 4).
- [ ] **Step 2: Add `.streamlit/config.toml`** with a theme (`primaryColor = "#2a6f97"`), and `[server] headless = true`.
- [ ] **Step 3: Push to GitHub.** Ask the user which GitHub account and whether the repo is public; then `gh repo create` and push. (Needs the user's confirmation, since this publishes the code.)
- [ ] **Step 4: Deploy on Streamlit Community Cloud** (user signs in at share.streamlit.io with GitHub; main file `app/Home.py`; Python 3.13). Open the live URL, click through every page, and put the URL in the README.
- [ ] **Step 5: Commit** `docs: README with run, regenerate and live link`.
