"""Caption text for the app pages, built from the reference results so numbers never go stale."""
import pandas as pd


def _row(table: list, model: str) -> dict:
    return next(r for r in table if r["Model"] == model)


def regression(p1: dict, ref: dict) -> tuple[pd.DataFrame, str]:
    reg = pd.DataFrame(p1["regression"])[["Model", "Features", "Target", "RMSE", "MAE", "R2"]].assign(Unit="2")
    gam = pd.DataFrame(ref["gam"]["table"]).rename(columns={"R²": "R2"}).assign(Unit="5", Features="raw",
                                                                              Target="cnt")
    df = pd.concat([reg, gam[reg.columns]]).sort_values("MAE").reset_index(drop=True)
    top = df.iloc[0]
    raw = reg[(reg.Features == "raw") & (reg.Target == "cnt")].MAE.min()
    g = _row(ref["gam"]["table"], "Poisson GAM (splines)")["MAE"]
    cap = (f"Lowest error: {top.Model} on {top.Features} features with a {top.Target} target "
           f"(MAE {top.MAE:.2f}). Linear models only get there with the hand-built hour × working-day features "
           f"and a log target; on raw features they reach MAE {raw:.0f}. The Poisson GAM learns the same shape "
           f"from raw inputs with smooth terms (MAE {g:.2f}).")
    return df, cap


def classification(p1: dict, ref: dict) -> tuple[pd.DataFrame, str]:
    c2 = pd.DataFrame(p1["classification"])
    c2 = c2.assign(Model=c2.Model + " (" + c2.Features + ")", Unit="2")
    c5 = pd.DataFrame(ref["cls"]).assign(Unit="5")
    cols = ["Unit", "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    df = pd.concat([c2[cols], c5[cols]]).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    top = df.iloc[0]
    best5 = df[df.Unit == "5"].iloc[0]
    raw2 = c2[c2.Model == "Logistic regression (L2) (raw)"].iloc[0]
    cap = (f"Top ROC-AUC: {top.Model} ({top['ROC-AUC']:.3f}, Unit {top.Unit}); the best Unit 5 model is "
           f"{best5.Model} ({best5['ROC-AUC']:.3f}). On raw features the ensembles are far ahead of linear models "
           f"(logistic regression {raw2['ROC-AUC']:.3f}); with hand-built hour × working-day features, the Unit 2 "
           f"models catch up. High demand means more than {p1['RES']['threshold']:.0f} rentals in the hour.")
    return df, cap


def forecast(ref: dict) -> str:
    k = _row(ref["forecast"], "Kalman LDS, 1 h ahead")["MAE"]
    p = _row(ref["forecast"], "Persistence (cnt[t-1])")["MAE"]
    y = _row(ref["forecast"], "Seasonal naive, same hour yesterday")["MAE"]
    h = _row(ref["forecast"], "HMM (K=4), 1 h ahead")["MAE"]
    return (f"Over the whole test period the 1-hour-ahead Kalman forecast has MAE {k:.1f} rentals, against "
            f"{p:.1f} for persistence and {y:.1f} for same-hour-yesterday ({min(p, y) / k:.1f}× lower error). "
            f"The HMM (MAE {h:.1f}) is a poor forecaster and is used for decoding demand states.")


def kmeans(ref: dict) -> str:
    k = ref["kmeans"]
    return (f"Silhouette ({k['sil']:.2f}) and Davies–Bouldin ({k['DB']:.2f}) both choose K = {k['K']}. The clusters "
            f"match the working-day flag almost exactly (ARI {k['ARI_workingday']:.2f}): commuters create morning "
            f"and evening peaks, leisure riders a midday hump. {len(ref['excluded_days'])} near-empty days "
            f"(e.g. Hurricane Sandy, 29 Oct 2012) are excluded.")


def pca(ref: dict) -> str:
    m = ref["kmeans_mismatch"]
    return (f"Commute days and leisure days form two well-separated groups along PC 1. Only {m['n']} days sit in "
            f"the group that does not match their calendar flag: {m['holidays']} public holidays that look like "
            f"commute days, and working days (mostly around Thanksgiving and Christmas) that look like leisure days.")


def gam_curve(ref: dict) -> str:
    return (f"The GAM is a sum of smooth effects; in the notebook, demand rises with temperature up to about "
            f"{ref['gam']['temp_peak_C']:.0f} °C, then falls.")
