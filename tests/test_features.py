"""Recorded-weather prediction, whole-day view, forecast horizon and the data explorer."""
from datetime import date

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import core

HOUR = pd.read_csv("data/hour.csv")
HOUR["date"] = pd.to_datetime(HOUR.dteday)
HOUR = HOUR.sort_values(["date", "hr"]).reset_index(drop=True)


def page(name):
    at = AppTest.from_file(f"../app/pages/{name}", default_timeout=120)
    at.run()
    return at


def test_recorded_conditions_match_csv():
    r = core.recorded_conditions(HOUR, date(2012, 8, 15), 8)
    raw = HOUR[(HOUR.dteday == "2012-08-15") & (HOUR.hr == 8)].iloc[0]
    assert r["cnt"] == raw.cnt and r["weathersit"] == raw.weathersit
    assert r["temp_c"] == pytest.approx(raw.temp * 41, abs=0.05)
    assert r["hum_pct"] == round(raw.hum * 100) and r["period"] == "test"
    assert core.recorded_conditions(HOUR, date(2011, 1, 1), 0)["period"] == "training"


def test_recorded_conditions_missing_hour_and_out_of_range():
    have = set(zip(HOUR.dteday, HOUR.hr))
    missing = next((d, h) for d in HOUR.dteday.unique() for h in range(24) if (d, h) not in have)
    assert core.recorded_conditions(HOUR, pd.Timestamp(missing[0]).date(), missing[1]) is None
    assert core.recorded_conditions(HOUR, date(2014, 5, 1), 8) is None


def test_same_hour_last_day_never_looks_ahead():
    cnt = np.arange(500.0)
    steps = np.arange(1, 49)
    t0 = 300
    base = core.same_hour_last_day(cnt, t0, steps)
    assert (base < t0).all()  # only hours before the forecast start
    assert base[0] == t0 - 24 and base[24] == t0 - 24  # 1 h and 25 h ahead both use the last full day


def test_predict_defaults_use_recorded_weather():
    at = page("1_Predict.py")
    assert not at.exception, at.exception
    assert at.toggle(key="use_rec").value is True
    labels = [m.label for m in at.metric]
    assert "Actual rentals (recorded)" in labels and "Rentals that day" in labels
    raw = HOUR[(HOUR.dteday == "2012-08-15") & (HOUR.hr == 8)].iloc[0]
    assert at.slider[0].value == pytest.approx(round(raw.temp * 41, 1))
    assert at.slider[0].disabled


def test_predict_manual_mode_and_future_date():
    at = page("1_Predict.py")
    at.toggle(key="use_rec").set_value(False)
    at.run()
    assert not at.exception and not at.slider[0].disabled
    at.date_input[0].set_value(date(2014, 5, 1))
    at.run()
    assert not at.exception, at.exception
    assert at.toggle(key="use_rec").disabled and len(at.info) >= 1


def test_forecast_48h_at_end():
    at = page("2_Forecast_24h.py")
    at.date_input[0].set_value(date(2012, 12, 30))
    at.select_slider(key="fc_horizon").set_value(48)
    at.run()
    assert not at.exception, at.exception


def test_explore_filters_and_empty_selection():
    at = page("5_Explore_Data.py")
    assert not at.exception, at.exception
    assert at.metric[0].value == "17,379"
    at.multiselect(key="ex_weather").set_value([4])
    at.radio(key="ex_dtype").set_value("Working days")
    at.run()
    assert not at.exception, at.exception
