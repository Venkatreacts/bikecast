"""Rebuild the demo-app models by executing the Units 3-5 notebook unchanged,
check its numbers against reference/results_units345.json, then export artifacts/.

Run from the repo root:  python training/train_models.py   (about 5 minutes)
"""
import json
import os
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Team8_Bike_Sharing_ML_PBL2_Units3-5.ipynb"
REF = ROOT / "reference" / "results_units345.json"
TOL = 1e-3  # reference tables are rounded to 4 decimals
PARITY_TEST_IDX = [0, 1, 2, 3, 4, 100, 2000]


def _ada_without_algorithm(*args, algorithm="SAMME", **kwargs):
    from sklearn.ensemble import AdaBoostClassifier

    assert algorithm == "SAMME", algorithm
    return AdaBoostClassifier(*args, **kwargs)


def run_notebook(nb_path: Path, workdir: Path) -> dict:
    """Execute every code cell of the notebook in one namespace, with cwd = workdir."""
    cells = json.loads(Path(nb_path).read_text(encoding="utf8"))["cells"]
    ns = {"__name__": "__notebook__", "display": lambda *a, **k: None}
    plt.show = lambda *a, **k: plt.close("all")
    old = os.getcwd()
    os.chdir(workdir)
    try:
        for i, cell in enumerate(c for c in cells if c["cell_type"] == "code"):
            src = "".join(cell["source"])
            src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
            exec(compile(src, f"<notebook cell {i}>", "exec"), ns)
            ns["display"] = lambda *a, **k: None  # a cell may import IPython's display
            if i == 0:  # scikit-learn >= 1.8 removed `algorithm`; SAMME is now the only (default) algorithm
                ns["AdaBoostClassifier"] = _ada_without_algorithm
    finally:
        os.chdir(old)
        plt.close("all")
    return ns


def _row(table: list, prefix: str) -> dict:
    return next(r for r in table if r["Model"].startswith(prefix))


def _checks(R: dict) -> dict:
    return {
        "kalman.phi": R["kalman"]["phi"],
        "kalman.Q": R["kalman"]["Q"],
        "kalman.R": R["kalman"]["R"],
        "forecast Kalman 1h MAE": _row(R["forecast"], "Kalman LDS, 1 h ahead")["MAE"],
        "Stacking ROC-AUC": _row(R["cls"], "Stacking")["ROC-AUC"],
        "Stacking Accuracy": _row(R["cls"], "Stacking")["Accuracy"],
        "GAM MAE": _row(R["gam"]["table"], "Poisson GAM (splines)")["MAE"],
        "kmeans silhouette": R["kmeans"]["sil"],
    }


def check_parity(R: dict, ref: dict) -> list[str]:
    """One message per headline number that differs from the reference by more than TOL."""
    got, want = _checks(R), _checks(ref)
    return [f"{k}: got {got[k]:.6f}, reference {want[k]:.6f}" for k in want if abs(got[k] - want[k]) > TOL]


def export(ns: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    train, test, n_train = ns["train"], ns["test"], ns["n_train"]

    joblib.dump(
        {
            "base": ns["fitted_base"],
            "meta": ns["meta"],
            "gam": ns["gam"],
            "thresh": float(ns["THRESH"]),
            "atemp_fit": np.polyfit(train.temp, train.atemp, 1).tolist(),
            "sklearn_version": sklearn.__version__,
        },
        out / "models.joblib",
        compress=3,
    )
    json.dump(
        {"phi": float(ns["phi_h"]), "Q": float(ns["Q_h"]), "R": float(ns["R_h"]),
         "profile": [float(v) for v in ns["m_tr"]], "P0": 1.0},
        open(out / "kalman.json", "w"), indent=1,
    )

    prof = ns["prof_km"]
    pd.DataFrame([{"cluster": k, "hr": h, "share": prof[k, h]} for k in range(len(prof)) for h in range(24)]
                 ).to_csv(out / "kmeans_profiles.csv", index=False)
    M, Zp = ns["M"], ns["Zp"]
    pd.DataFrame({"date": M.index.date, "cluster": ns["lab_km"], "workingday": ns["dinfo"].workingday.values,
                  "total": ns["tot"].values, "pc1": Zp[:, 0], "pc2": Zp[:, 1]}).to_csv(out / "days.csv", index=False)

    st = ns["path_te"]
    pd.DataFrame({"date": test.date.dt.date.values, "hr": test.hr.values, "cnt": test.cnt.values, "state": st,
                  "state_cnt": np.expm1(ns["hm"].mu[st])}).to_csv(out / "hmm_test.csv", index=False)

    p_stack = next(v for k, v in ns["PROB"].items() if k.startswith("Stacking"))
    pg = ns["pg"]
    z1, Pp, z24, v24, R_h = ns["z1"], ns["Pp_all"], ns["z24"], ns["v24"], ns["R_h"]
    json.dump(
        {
            "test_idx": PARITY_TEST_IDX,
            "p_stack": [float(p_stack[i]) for i in PARITY_TEST_IDX],
            "gam": [float(pg[i]) for i in PARITY_TEST_IDX],
            "kalman": [{"t0": int(t), "z1": float(z1[t]), "var1": float(Pp[t] + R_h),
                        "z24": float(z24[t + 23]), "var24": float(v24[t + 23])}
                       for t in (n_train + 100, n_train + 2000)],
        },
        open(out / "parity.json", "w"), indent=1,
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # notebook prints arrows; Windows console default is cp1252
    build = ROOT / "build" / "notebook_run"
    build.mkdir(parents=True, exist_ok=True)
    for f in ("hour.csv", "day.csv"):
        shutil.copy(ROOT / "data" / f, build / f)
    ns = run_notebook(NOTEBOOK, build)
    produced = json.load(open(build / "results_units345.json", encoding="utf8"))
    msgs = check_parity(produced, json.load(open(REF, encoding="utf8")))
    if msgs:
        print("PARITY FAILED - artifacts not written:")
        print("\n".join("  " + m for m in msgs))
        return 1
    export(ns, ROOT / "artifacts")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
