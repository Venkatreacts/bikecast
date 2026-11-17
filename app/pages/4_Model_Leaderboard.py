import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

import ui  # noqa: E402
from app import captions  # noqa: E402

ui.setup("Model leaderboard")
st = ui.st
art = ui.get_artifacts()
ref, p1 = art["ref"], art["ref_pbl1"]
st.markdown("All models were trained on the first 80 % of hours and tested on the last 20 % "
            "(7 Aug – 31 Dec 2012). Numbers are read from the notebooks' saved results.")

st.subheader("Predicting the hourly count (Units 2 and 5)")
reg, cap = captions.regression(p1, ref)
st.dataframe(reg.round(3), hide_index=True)
st.caption(cap)

st.subheader("High vs low demand classification (Units 2 and 5)")
cls, cap = captions.classification(p1, ref)
st.dataframe(cls.round(3), hide_index=True)
st.caption(cap)

st.subheader("Forecasting the next hours (Unit 4)")
st.dataframe(pd.DataFrame(ref["forecast"]).round(3), hide_index=True)
st.caption(captions.forecast(ref))

st.subheader("Choosing the number of clusters (Unit 3)")
st.dataframe(pd.DataFrame(ref["kmeans_table"]).round(3), hide_index=True)
st.caption(captions.kmeans(ref))
