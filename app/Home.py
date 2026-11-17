import ui

ui.setup("Bike-Sharing Demand Forecasting for Smart Urban Mobility")
st = ui.st

st.caption("21CSC305P Machine Learning · Team 8 · B.Tech CSE (AI & ML) · SRMIST Kattankulathur")

st.markdown(
    "Bike-sharing operators must place bikes and staff before riders arrive. "
    "Using two years of hourly rentals from Washington D.C.'s Capital Bikeshare (UCI dataset 275, 17,379 hours), "
    "we model how weather, time and the calendar drive demand, and turn that into three working tools: "
    "an hourly demand predictor, a 24-hour forecast, and a map of typical demand patterns."
)

st.subheader("Team")
st.table(
    {
        "Name": ["Ali Hasan Ansari", "Fayas K N", "Konanki Revanth Chowdary", "Venkatanathan O R V", "Dhruva Mishra"],
        "Register number": ["RA2411026011160", "RA2411026011161", "RA2411026011162", "RA2411026011163",
                            "RA2411026011164"],
    }
)

st.subheader("How the project fits together")
st.graphviz_chart(
    """
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef4f8", fontname="Helvetica"];
  data [label="UCI hourly rentals\\n2011-2012"];
  prep [label="Preprocessing\\nleakage removal, chronological 80/20 split"];
  u12 [label="Units 1-2\\nprobability, regression,\\nBayesian logistic, SVM"];
  u3 [label="Unit 3\\nK-means, GMM, hierarchical,\\nPCA, factor analysis"];
  u4 [label="Unit 4\\nHMM, Viterbi,\\nKalman filter"];
  u5 [label="Unit 5\\nCART, RF, boosting,\\nstacking, GAM"];
  app [label="This app", fillcolor="#2a6f97", fontcolor="white"];
  data -> prep -> {u12 u3 u4 u5}; {u3 u4 u5} -> app; u12 -> app [style=dashed, label="leaderboard"];
}
"""
)

st.subheader("Pages")
st.markdown(
    "- **Predict** (Unit 5): enter a date, hour and weather; get expected rentals (GAM) and the probability of "
    "high demand (stacking ensemble).\n"
    "- **Forecast 24h** (Unit 4): pick any hour in the test period; the Kalman filter forecasts the next 24 hours "
    "with a 95 % band, compared against what actually happened.\n"
    "- **Demand Patterns** (Units 3-4): commute vs leisure day shapes (K-means), the PCA map of days, and hidden "
    "demand states decoded by the HMM.\n"
    "- **Model Leaderboard** (Units 1-5): every model's test metrics in one place."
)
st.caption("All numbers come from the project notebooks, re-run and checked by `training/train_models.py`.")
