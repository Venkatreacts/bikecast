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
            "(7 Aug – 31 Dec 2012). Numbers are read from the notebooks' saved results. Click a column "
            "header to sort; each table can be downloaded as CSV.")


def table(df: pd.DataFrame, name: str) -> None:
    st.dataframe(df.round(3), hide_index=True)
    st.download_button(f"Download {name} (CSV)", df.to_csv(index=False).encode(), f"{name}.csv", "text/csv",
                       key=f"dl_{name}")

st.subheader("Predicting the hourly count (Units 2 and 5)")
reg, cap = captions.regression(p1, ref)
table(reg, "regression_results")
st.caption(cap)

st.subheader("High vs low demand classification (Units 2 and 5)")
cls, cap = captions.classification(p1, ref)
table(cls, "classification_results")
st.caption(cap)

st.subheader("Forecasting the next hours (Unit 4)")
table(pd.DataFrame(ref["forecast"]), "forecast_results")
st.caption(captions.forecast(ref))

st.subheader("Choosing the number of clusters (Unit 3)")
table(pd.DataFrame(ref["kmeans_table"]), "kmeans_selection")
st.caption(captions.kmeans(ref))
