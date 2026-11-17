import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

import ui  # noqa: E402

ui.setup("Model leaderboard")
st = ui.st
art = ui.get_artifacts()
ref, p1 = art["ref"], art["ref_pbl1"]
st.markdown("All models were trained on the first 80 % of hours and tested on the last 20 % "
            "(7 Aug – 31 Dec 2012). Numbers are read from the notebooks' saved results.")

st.subheader("Predicting the hourly count (Units 2 and 5)")
reg = pd.DataFrame(p1["regression"])[["Model", "Features", "Target", "RMSE", "MAE", "R2"]].assign(Unit="2")
gam = pd.DataFrame(ref["gam"]["table"]).rename(columns={"R²": "R2"}).assign(Unit="5", Features="raw", Target="cnt")
st.dataframe(pd.concat([reg, gam[reg.columns]]).sort_values("MAE").round(3), hide_index=True)
st.caption("Nonlinear models win clearly: the Poisson GAM (MAE 44.9) matches gradient boosting while staying "
           "interpretable; linear models on raw features miss the hour × working-day interaction.")

st.subheader("High vs low demand classification (Units 2 and 5)")
c2 = pd.DataFrame(p1["classification"])
c2 = c2.assign(Model=c2.Model + " (" + c2.Features + ")", Unit="2")
c5 = pd.DataFrame(ref["cls"]).assign(Unit="5")
cols = ["Unit", "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
st.dataframe(pd.concat([c2[cols], c5[cols]]).sort_values("ROC-AUC", ascending=False).round(3), hide_index=True)
st.caption("Tree ensembles beat every Unit 2 classifier; stacking reaches ROC-AUC 0.978 and accuracy 0.927. "
           "High demand means more than 130 rentals in the hour (the training median).")

st.subheader("Forecasting the next hours (Unit 4)")
st.dataframe(pd.DataFrame(ref["forecast"]).round(3), hide_index=True)
st.caption("The Kalman filter's 1-hour forecast is almost 3× more accurate than naive baselines; the HMM is "
           "kept for interpreting demand states, not for forecasting.")

st.subheader("Choosing the number of clusters (Unit 3)")
st.dataframe(pd.DataFrame(ref["kmeans_table"]).round(3), hide_index=True)
st.caption("Highest silhouette and lowest Davies–Bouldin both occur at K = 2.")
