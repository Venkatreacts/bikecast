# Bike-Sharing Demand Forecasting for Smart Urban Mobility

21CSC305P Machine Learning project, Team 8, B.Tech CSE (AI & ML), SRMIST Kattankulathur.

| Name | Register number |
|---|---|
| Ali Hasan Ansari | RA2411026011160 |
| Fayas K N | RA2411026011161 |
| Konanki Revanth Chowdary | RA2411026011162 |
| Venkatanathan O R V | RA2411026011163 |
| Dhruva Mishra | RA2411026011164 |

Live demo: https://bikecast.streamlit.app

## What is here

- `notebooks/`: the full analysis. PBL 1 covers Units 1–2 (probability, regression, Bayesian logistic regression, SVM). PBL 2/3 covers Units 3–5 (clustering, PCA, HMM, Kalman filter, trees, boosting, stacking, GAM).
- `app/`: a Streamlit app with four pages:
  - **Predict**: hourly rentals and the probability of high demand.
  - **Forecast 24h**: a Kalman filter forecast for the next 24 hours.
  - **Demand Patterns**: K-means day shapes, a PCA map and the hidden HMM states.
  - **Model Leaderboard**: the test metrics of every model.
- `training/train_models.py`: re-runs the Units 3–5 notebook unchanged. It checks the notebook's headline numbers against `reference/results_units345.json`, then saves the models the app uses to `artifacts/`.
- `tests/`: checks that the app gives exactly the notebook's predictions and that every page runs.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/Home.py
```

## Regenerate the models (about 10 minutes)

```bash
python training/train_models.py
```

It prints `OK` only when every checked number matches the notebook's saved results.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

## Data

The course batch sheet (`data/ML_Project_Batch_details.xlsx`, read by the PBL 1 notebook to confirm the team
rows) is not published because it lists the whole class. Place it in `data/` to re-run that cell.

Fanaee-T, H. and Gama, J. (2013). *Bike Sharing Dataset*. UCI Machine Learning Repository,
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset (CC BY 4.0).
Hourly and daily rental counts from Capital Bikeshare, Washington D.C., 2011–2012.
