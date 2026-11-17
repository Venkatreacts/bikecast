"""All computation for the demo app. Pages only call these functions.

Every formula here mirrors the Units 3-5 notebook; tests/test_core.py checks
the results against values exported from the notebook run (artifacts/parity.json).
"""
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RAW_COLS = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit",
            "temp", "atemp", "hum", "windspeed"]
BASE_NAMES = ["cart", "rf", "gb", "ada"]


class ArtifactVersionError(RuntimeError):
    def __init__(self, saved: str, installed: str):
        super().__init__(f"Models were saved with scikit-learn {saved}, but scikit-learn {installed} is installed. "
                         f"Install the pinned versions: pip install -r requirements.txt")


def load_artifacts() -> dict:
    saved = json.load(open(ART / "meta.json"))["sklearn_version"]  # read before unpickling, which may crash
    if saved != sklearn.__version__:
        raise ArtifactVersionError(saved, sklearn.__version__)
    try:
        models = joblib.load(ART / "models.joblib")
    except Exception as e:
        raise ArtifactVersionError(saved, sklearn.__version__) from e
    hour = pd.read_csv(ROOT / "data" / "hour.csv")
    hour["date"] = pd.to_datetime(hour.dteday)
    hour = hour.sort_values(["date", "hr"]).reset_index(drop=True)
    return {
        "models": models,
        "kalman": json.load(open(ART / "kalman.json")),
        "kmeans_profiles": pd.read_csv(ART / "kmeans_profiles.csv"),
        "days": pd.read_csv(ART / "days.csv", parse_dates=["date"]),
        "hmm_test": pd.read_csv(ART / "hmm_test.csv", parse_dates=["date"]),
        "hour": hour,
        "day": pd.read_csv(ROOT / "data" / "day.csv", parse_dates=["dteday"]),
        "ref": json.load(open(ROOT / "reference" / "results_units345.json", encoding="utf8")),
        "ref_pbl1": json.load(open(ROOT / "reference" / "results.json", encoding="utf8")),
    }


def _season(d: date) -> int:
    md = (d.month, d.day)
    if (3, 21) <= md < (6, 21):
        return 2
    if (6, 21) <= md < (9, 23):
        return 3
    if (9, 23) <= md < (12, 21):
        return 4
    return 1


def calendar_fields(d: date, day_df: pd.DataFrame) -> dict:
    out_of_range = d.year not in (2011, 2012)
    if out_of_range:
        holiday = int(len(USFederalHolidayCalendar().holidays(start=d, end=d)) > 0)
    else:
        holiday = int(day_df.loc[day_df.dteday.dt.date == d, "holiday"].iloc[0])
    weekday = (d.weekday() + 1) % 7  # dataset: 0 = Sunday
    return {"season": _season(d), "yr": 0 if d.year == 2011 else 1, "mnth": d.month, "holiday": holiday,
            "weekday": weekday, "workingday": int(weekday not in (0, 6) and not holiday),
            "out_of_range": out_of_range}


def make_row(d: date, hr: int, weathersit: int, temp_c: float, hum_pct: float, wind_kmh: float,
             day_df: pd.DataFrame, atemp_fit: list) -> pd.DataFrame:
    c = calendar_fields(d, day_df)
    temp = temp_c / 41
    row = {**{k: c[k] for k in ("season", "yr", "mnth", "holiday", "weekday", "workingday")},
           "hr": hr, "weathersit": weathersit, "temp": temp, "atemp": float(np.polyval(atemp_fit, temp)),
           "hum": hum_pct / 100, "windspeed": wind_kmh / 67}
    return pd.DataFrame([row])[RAW_COLS]


def _gam_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Same columns as the notebook's gdat()."""
    d = rows[["temp", "hum", "windspeed", "season", "weathersit", "weekday", "mnth", "yr", "holiday",
              "workingday"]].copy()
    d["hr_w"] = np.where(rows.workingday == 1, rows.hr, -1)
    d["hr_nw"] = np.where(rows.workingday == 0, rows.hr, -1)
    return d


def predict(models: dict, rows: pd.DataFrame) -> pd.DataFrame:
    X = rows[RAW_COLS].astype(float).values
    base = np.column_stack([m.predict_proba(X)[:, 1] for m in models["base"]])
    pc = np.clip(base, 1e-4, 1 - 1e-4)
    p_stack = models["meta"].predict_proba(np.log(pc / (1 - pc)))[:, 1]
    out = pd.DataFrame(base, columns=[f"p_{n}" for n in BASE_NAMES])
    out["p_stack"] = p_stack
    out["high"] = p_stack >= 0.5
    out["count_gam"] = models["gam"].predict(_gam_frame(rows.reset_index(drop=True)))
    return out


def kalman_forecast(cnt: np.ndarray, workingday: np.ndarray, hr: np.ndarray, t0: int, k: dict,
                    horizon: int = 24) -> pd.DataFrame:
    """Filter the profile residual over rows 0..t0-1, then forecast rows t0..t0+horizon-1."""
    phi, Q, R, prof = k["phi"], k["Q"], k["R"], np.asarray(k["profile"])
    cell = hr * 2 + workingday
    r = np.log1p(cnt.astype(float)) - prof[cell]
    l, P = 0.0, k["P0"]
    for t in range(t0):
        lp, Pp = phi * l, phi ** 2 * P + Q
        K = Pp / (Pp + R)
        l, P = lp + K * (r[t] - lp), (1 - K) * Pp
    rows = []
    for h in range(1, horizon + 1):
        idx = t0 - 1 + h
        while idx >= len(cell):  # shortcut: past the data end, reuse the same hour one day earlier
            idx -= 24
        z = prof[cell[idx]] + phi ** h * l
        var = phi ** (2 * h) * P + Q * (1 - phi ** (2 * h)) / (1 - phi ** 2) + R
        s = 1.96 * np.sqrt(var)
        rows.append({"step": h, "mean": np.expm1(z), "lo95": np.expm1(z - s), "hi95": np.expm1(z + s)})
    return pd.DataFrame(rows)
