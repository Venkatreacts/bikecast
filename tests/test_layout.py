import json

import pandas as pd
import pytest


def test_data_present():
    hour = pd.read_csv("data/hour.csv")
    day = pd.read_csv("data/day.csv")
    assert hour.shape == (17379, 17) and day.shape == (731, 16)
    ref = json.load(open("reference/results_units345.json", encoding="utf8"))
    assert ref["kalman"]["phi"] == pytest.approx(0.93493, abs=1e-5)
