"""Tests for the final-review fix pass: captions must state what the tables show,
version errors must be caught before unpickling, and the storm option must warn."""
import json

import pytest
from streamlit.testing.v1 import AppTest

from app import captions, core

REF = json.load(open("reference/results_units345.json", encoding="utf8"))
P1 = json.load(open("reference/results.json", encoding="utf8"))


def test_classification_caption_names_true_leader():
    df, cap = captions.classification(P1, REF)
    top = df.iloc[0]
    assert top.Model == "SVM RBF kernel (encoded)"
    assert "SVM RBF kernel (encoded)" in cap and f"{top['ROC-AUC']:.3f}" in cap
    assert "beat every Unit 2" not in cap


def test_regression_caption_names_true_leader():
    df, cap = captions.regression(P1, REF)
    top = df.iloc[0]
    assert top.Model == "Bayesian linear (evidence)" and f"{top.MAE:.2f}" in cap
    assert "win clearly" not in cap


def test_forecast_caption_from_ref():
    cap = captions.forecast(REF)
    rows = {r["Model"]: r["MAE"] for r in REF["forecast"]}
    for name in ["Kalman LDS, 1 h ahead", "Persistence (cnt[t-1])", "Seasonal naive, same hour yesterday"]:
        assert f"{rows[name]:.1f}" in cap


def test_kmeans_caption_from_ref():
    cap = captions.kmeans(REF) + captions.pca(REF)
    assert f"{REF['kmeans_mismatch']['n']} days" in cap and "99.9" not in cap
    assert f"{len(REF['excluded_days'])} near-empty days" in cap


def test_gam_caption_from_ref():
    assert f"{REF['gam']['temp_peak_C']:.0f} °C" in captions.gam_curve(REF)


def test_unpickle_failure_becomes_version_error(monkeypatch):
    def boom(*a, **k):
        raise AttributeError("Can't get attribute '_SomeInternal'")

    monkeypatch.setattr(core.joblib, "load", boom)
    with pytest.raises(core.ArtifactVersionError) as e:
        core.load_artifacts()
    assert "1.9.0" in str(e.value)


def test_storm_option_warns():
    at = AppTest.from_file("../app/pages/1_Predict.py", default_timeout=120)
    at.run()
    at.selectbox[1].set_value(4)
    at.run()
    assert not at.exception and len(at.warning) >= 1


def test_home_does_not_overclaim():
    at = AppTest.from_file("../app/Home.py", default_timeout=120)
    at.run()
    text = " ".join(c.value for c in at.caption)
    assert "All numbers come from the project notebooks, re-run" not in text
    assert "Units 3–5" in text
