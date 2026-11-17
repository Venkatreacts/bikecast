import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ui  # noqa: E402
from app import core  # noqa: E402

ui.setup("24-hour demand forecast (Kalman filter)")
st = ui.st
art = ui.get_artifacts()
hour, k, ref = art["hour"], art["kalman"], art["ref"]

st.markdown(
    "The filter tracks how far demand is running above or below the usual hour × working-day profile, "
    "then projects that gap forward. It only sees hours **before** the chosen start, as it would in real use."
)
c1, c2 = st.columns(2)
d = c1.date_input("Forecast start date (test period)", date(2012, 10, 17),
                  min_value=date(2012, 8, 7), max_value=date(2012, 12, 31))
h0 = c2.selectbox("Start hour", list(range(24)), index=6, key="fc_hour", format_func=lambda h: f"{h:02d}:00")

stamp = pd.Timestamp(d) + pd.Timedelta(hours=h0)
row_time = hour.date + pd.to_timedelta(hour.hr, unit="h")
t0 = min(int(np.searchsorted(row_time.values, stamp.to_datetime64())), len(hour) - 1)
st.caption(f"First forecast hour: {row_time.iloc[t0]:%d %b %Y %H:00} (hours missing from the data are skipped, "
           "exactly as in the notebook).")

cnt = hour.cnt.values.astype(float)
fc = core.kalman_forecast(cnt, hour.workingday.values, hour.hr.values, t0, k)
steps = fc.step.values
idx = t0 - 1 + steps
has = idx < len(cnt)
actual = np.where(has, cnt[np.minimum(idx, len(cnt) - 1)], np.nan)
persist = np.full(len(steps), cnt[t0 - 1])
yesterday = cnt[idx - 24]

lines = pd.concat([
    pd.DataFrame({"step": steps, "rentals": v, "series": s})
    for s, v in [("Actual", actual), ("Kalman forecast", fc["mean"].values),
                 ("Persistence (last hour)", persist), ("Same hour yesterday", yesterday)]
]).dropna()
band = alt.Chart(fc, title="Next 24 hours: forecast vs what actually happened").mark_area(
    opacity=0.2, color="#2a6f97").encode(x=alt.X("step:Q", title="hours ahead"), y="lo95:Q", y2="hi95:Q")
ln = alt.Chart(lines).mark_line(point=True).encode(
    x="step:Q", y=alt.Y("rentals:Q", title="rentals per hour"), color=alt.Color("series:N", scale=alt.Scale(
        domain=["Actual", "Kalman forecast", "Persistence (last hour)", "Same hour yesterday"],
        range=["#222222", "#2a6f97", "#e07a5f", "#81b29a"])),
    strokeDash=alt.condition(alt.datum.series == "Actual", alt.value([1, 0]), alt.value([4, 3])))
st.altair_chart(band + ln, width="stretch")
st.caption("Shaded band = Kalman 95 % prediction interval. Baselines use only information available at the start.")

if has.any():
    a = actual[has]
    mae = pd.DataFrame({"Method": ["Kalman forecast", "Persistence (last hour)", "Same hour yesterday"],
                        f"MAE over these {int(has.sum())} hours": [np.abs(a - p[has]).mean() for p in
                                                                   (fc["mean"].values, persist, yesterday)]})
    st.dataframe(mae.round(1), hide_index=True)
else:
    st.info("No actual counts exist after this point; only the forecast is shown.")

full = pd.DataFrame(ref["forecast"])[["Model", "MAE", "RMSE", "R²"]]
st.subheader("Whole test period (7 Aug – 31 Dec 2012)")
st.dataframe(full.round(2), hide_index=True)
st.caption("Over the full test period the 1-hour-ahead Kalman forecast has MAE 31.2 rentals, against 85.2 for "
           "persistence and 80.5 for same-hour-yesterday; the HMM is a poor forecaster and is used for decoding.")
