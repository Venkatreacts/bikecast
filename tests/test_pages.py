from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["app/Home.py", "app/pages/1_Predict.py", "app/pages/2_Forecast_24h.py",
         "app/pages/3_Demand_Patterns.py", "app/pages/4_Model_Leaderboard.py"]


def run(path):
    at = AppTest.from_file(f"../{path}", default_timeout=120)  # resolved relative to this file
    at.run()
    return at


@pytest.mark.parametrize("path", PAGES)
def test_page_runs(path):
    at = run(path)
    assert not at.exception, at.exception


def test_predict_shows_metrics():
    assert len(run("app/pages/1_Predict.py").metric) >= 2


def test_forecast_last_hour():
    at = run("app/pages/2_Forecast_24h.py")
    at.date_input[0].set_value(date(2012, 12, 31))
    at.selectbox(key="fc_hour").set_value(23)
    at.run()
    assert not at.exception, at.exception


def test_leaderboard_tables():
    assert len(run("app/pages/4_Model_Leaderboard.py").dataframe) >= 4
