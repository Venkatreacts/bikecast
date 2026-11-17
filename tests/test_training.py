import copy
import json
import os

from training.train_models import check_parity

REF = json.load(open("reference/results_units345.json", encoding="utf8"))


def test_check_parity_identical():
    assert check_parity(REF, REF) == []


def test_artifacts_load():
    import joblib
    import pandas as pd

    for f in ["models.joblib", "kalman.json", "kmeans_profiles.csv", "days.csv", "hmm_test.csv", "parity.json"]:
        assert os.path.exists(f"artifacts/{f}"), f
    assert len(joblib.load("artifacts/models.joblib")["base"]) == 4
    assert len(json.load(open("artifacts/kalman.json"))["profile"]) == 48
    days = pd.read_csv("artifacts/days.csv")
    assert len(days) == 728 and days.cluster.nunique() == 2


def test_check_parity_detects_drift():
    drifted = copy.deepcopy(REF)
    row = next(r for r in drifted["cls"] if r["Model"].startswith("Stacking"))
    row["ROC-AUC"] = 0.90
    msgs = check_parity(drifted, REF)
    assert len(msgs) == 1 and "ROC-AUC" in msgs[0]
