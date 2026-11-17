import itertools
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app import core

DAY = pd.read_csv("data/day.csv", parse_dates=["dteday"])
PARITY = json.load(open("artifacts/parity.json"))


@pytest.fixture(scope="module")
def art():
    return core.load_artifacts()


def test_calendar_matches_day_csv():
    for r in DAY.itertuples():
        got = core.calendar_fields(r.dteday.date(), DAY)
        want = dict(season=r.season, yr=r.yr, mnth=r.mnth, holiday=r.holiday, weekday=r.weekday, workingday=r.workingday)
        assert {k: got[k] for k in want} == want, r.dteday


def test_calendar_edges():
    c = core.calendar_fields(date(2012, 7, 4), DAY)
    assert (c["holiday"], c["workingday"], c["weekday"]) == (1, 0, 3)
    c = core.calendar_fields(date(2025, 3, 10), DAY)
    assert (c["yr"], c["out_of_range"], c["workingday"]) == (1, True, 1)


def test_predict_parity(art):
    hour = art["hour"]
    n_train = int(0.8 * len(hour))
    rows = hour.iloc[[n_train + i for i in PARITY["test_idx"]]]
    out = core.predict(art["models"], rows)
    assert out.p_stack.values == pytest.approx(PARITY["p_stack"], abs=1e-9)
    assert out.count_gam.values == pytest.approx(PARITY["gam"], abs=1e-9)


def test_kalman_parity(art):
    h = art["hour"]
    for e in PARITY["kalman"]:
        fc = core.kalman_forecast(h.cnt.values, h.workingday.values, h.hr.values, e["t0"], art["kalman"])
        assert fc["mean"].iloc[0] == pytest.approx(np.expm1(e["z1"]), rel=1e-9)
        assert fc["mean"].iloc[23] == pytest.approx(np.expm1(e["z24"]), rel=1e-9)
        assert fc["hi95"].iloc[23] == pytest.approx(np.expm1(e["z24"] + 1.96 * np.sqrt(e["var24"])), rel=1e-9)


def test_extreme_inputs(art):
    for t, hum, wind in itertools.product([-10, 45], [0, 100], [0, 67]):
        row = core.make_row(date(2012, 8, 15), 8, 1, t, hum, wind, DAY, art["models"]["atemp_fit"])
        out = core.predict(art["models"], row).iloc[0]
        assert np.isfinite(out.count_gam) and out.count_gam >= 0
        assert 0 <= out.p_stack <= 1


def test_forecast_at_end(art):
    h = art["hour"]
    fc = core.kalman_forecast(h.cnt.values, h.workingday.values, h.hr.values, len(h) - 1, art["kalman"])
    assert len(fc) == 24 and np.isfinite(fc[["mean", "lo95", "hi95"]].values).all()


def test_version_mismatch(monkeypatch):
    monkeypatch.setattr(core.sklearn, "__version__", "0.0")
    with pytest.raises(core.ArtifactVersionError) as e:
        core.load_artifacts()
    assert "0.0" in str(e.value) and "1.9.0" in str(e.value)
