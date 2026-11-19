import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ui  # noqa: E402
from app import captions, core  # noqa: E402

ui.setup("Predict hourly demand")
st = ui.st
art = ui.get_artifacts()
models, day, hour = art["models"], art["day"], art["hour"]
WEATHER = core.WEATHER
SEASON = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
thresh = models["thresh"]

c1, c2, c3 = st.columns(3)
d = c1.date_input("Date", date(2012, 8, 15))
hr = c1.selectbox("Hour of day", list(range(24)), index=8, format_func=lambda h: f"{h:02d}:00")
rec = core.recorded_conditions(hour, d, hr)
in_range = d.year in (2011, 2012)

use_rec = c1.toggle("Use recorded weather for this hour", value=rec is not None, disabled=rec is None, key="use_rec",
                    help="2011–2012 hours have their real weather in the dataset. Turn off to try your own conditions.")
use_rec = bool(use_rec and rec is not None)
base = rec or {"weathersit": 1, "temp_c": 24.0, "hum_pct": 60, "wind_kmh": 12.0}
k = f"{'rec' if use_rec else 'man'}_{d}_{hr}"  # new key per date/hour/mode, so inputs reset to that hour's values
w = c2.selectbox("Weather", list(WEATHER), index=base["weathersit"] - 1, format_func=WEATHER.get,
                 disabled=use_rec, key=f"w_{k}")
temp_c = c2.slider("Temperature (°C)", -10.0, 45.0, float(base["temp_c"]), 0.5, disabled=use_rec, key=f"t_{k}")
hum = c3.slider("Humidity (%)", 0, 100, int(base["hum_pct"]), disabled=use_rec, key=f"h_{k}")
wind = c3.slider("Wind speed (km/h)", 0.0, 67.0, float(base["wind_kmh"]), 0.5, disabled=use_rec, key=f"v_{k}")

if in_range and rec is None:
    c1.caption("No rentals were logged in this hour, so the dataset has no weather for it. Enter conditions by hand.")
elif use_rec:
    c1.caption("Weather inputs show what was recorded. Turn the toggle off to change them.")

row = core.make_row(d, hr, w, temp_c, hum, wind, day, models["atemp_fit"])
out = core.predict(models, row).iloc[0]
cal = core.calendar_fields(d, day)
if w == 4:
    n_storm = int((hour.iloc[: core.N_TRAIN].weathersit == 4).sum())
    st.warning(f"Only {n_storm} storm hours exist in the training data, so predictions for this weather are "
               "unreliable (the models can even predict more rentals than for light rain).")
if cal["out_of_range"]:
    st.info("The models were trained on 2011–2012 only. This date is treated as 2012 (the later, busier year).")

day_type = "Working day" if cal["workingday"] else ("Holiday" if cal["holiday"] else "Weekend")
m1, m2, m3, m4 = st.columns(4)
err = out.count_gam - rec["cnt"] if use_rec else None
m1.metric("Predicted rentals this hour (GAM)", f"{out.count_gam:,.0f}",
          None if err is None else f"{err:+,.0f} vs actual", delta_color="off")
m2.metric("High-demand probability (stacking)", f"{out.p_stack:.2f}",
          f"{'HIGH' if out.high else 'LOW'} (> {thresh:.0f} rides = high)", delta_color="off")
if rec is not None:
    m3.metric("Actual rentals (recorded)", f"{rec['cnt']:,}",
              f"{rec['registered']:,} registered · {rec['casual']:,} casual", delta_color="off")
    if use_rec:
        actual_high = rec["cnt"] > thresh
        st.caption(f"Actual label: {'HIGH' if actual_high else 'LOW'} · the classifier was "
                   f"{'right' if actual_high == bool(out.high) else 'wrong'} for this hour.")
else:
    m3.metric("Actual rentals (recorded)", "not in data")
m4.metric("Day type", day_type)

# ---- the chosen day at a glance (only dates in the dataset)
dh = core.day_hours(hour, d)
if len(dh):
    days = art["days"]
    wd_share = days.groupby("cluster").workingday.mean()
    drow = days[days.date.dt.date == d]
    shape = ("Commute shape" if wd_share[drow.cluster.iloc[0]] > 0.5 else "Leisure shape") if len(drow) else \
        "Not clustered"
    split = pd.Timestamp("2012-08-07")
    period = "Training" if pd.Timestamp(d) < split else ("Test" if pd.Timestamp(d) > split else "Split day")
    with st.container(border=True):
        st.markdown(f"**{d:%A %d %B %Y}** · {SEASON[cal['season']]} · {day_type}")
        a, b, c, e = st.columns(4)
        a.metric("Rentals that day", f"{int(dh.cnt.sum()):,}")
        b.metric("Registered riders", f"{dh.registered.sum() / max(dh.cnt.sum(), 1):.0%}")
        c.metric("Day shape (K-means, Unit 3)", shape.replace(" shape", ""))
        e.metric("Data split", period)
        notes = {"Training": "the models learned from this day",
                 "Test": "the models never saw this day while training",
                 "Split day": "training until 11:00, test from 12:00"}
        extra = " Near-empty days (under 500 rentals) were left out of clustering." if shape == "Not clustered" else ""
        st.caption(f"{period}: {notes[period]}. Temperature ranged {dh.temp_c.min():.0f}–{dh.temp_c.max():.0f} °C; "
                   f"worst weather: {WEATHER[int(dh.weathersit.max())].lower()}.{extra}")

    rows = pd.concat([core.make_row(d, int(r.hr), int(r.weathersit), r.temp_c, r.hum_pct, r.wind_kmh, day,
                                    models["atemp_fit"]) for r in dh.itertuples()])
    pr = core.predict(models, rows)
    whole = pd.DataFrame({"hr": dh.hr.values, "Predicted (GAM)": pr.count_gam.values, "Actual": dh.cnt.values})
    long = whole.melt("hr", var_name="series", value_name="rentals")
    st.subheader("The whole day: predicted vs actual, with the recorded weather")
    lines = alt.Chart(long).mark_line(point=True).encode(
        x=alt.X("hr:Q", title="hour of day", scale=alt.Scale(domain=[0, 23])),
        y=alt.Y("rentals:Q", title="rentals per hour"),
        color=alt.Color("series:N", title=None, scale=alt.Scale(domain=["Actual", "Predicted (GAM)"],
                                                                range=["#222222", "#2a6f97"])),
        tooltip=["hr:Q", "series:N", alt.Tooltip("rentals:Q", format=",.0f")])
    marker = alt.Chart(pd.DataFrame({"hr": [hr]})).mark_rule(color="#e07a5f", strokeDash=[4, 3]).encode(x="hr:Q")
    thr = alt.Chart(pd.DataFrame({"y": [thresh]})).mark_rule(color="#999999", strokeDash=[2, 2]).encode(y="y:Q")
    st.altair_chart(lines + marker + thr, width="stretch")
    mae = np.abs(whole["Predicted (GAM)"] - whole.Actual).mean()
    hits = int(((pr.p_stack.values >= 0.5) == (dh.cnt.values > thresh)).sum())
    st.caption(f"Orange dashed line = selected hour; grey dotted = {thresh:.0f}-ride threshold. GAM error over this "
               f"day: MAE {mae:,.1f} rentals. The stacking classifier labelled {hits} of {len(dh)} hours correctly.")
    pcls = pd.DataFrame({"hr": dh.hr.values, "P(high)": pr.p_stack.values,
                         "actual": np.where(dh.cnt.values > thresh, "actually high", "actually low")})
    st.altair_chart(alt.Chart(pcls, title="Stacking probability of high demand, hour by hour").mark_bar().encode(
        x=alt.X("hr:O", title="hour of day"), y=alt.Y("P(high):Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("actual:N", title=None, scale=alt.Scale(domain=["actually high", "actually low"],
                                                                 range=["#2a6f97", "#c9d6df"])),
        tooltip=["hr:O", alt.Tooltip("P(high):Q", format=".2f"), "actual:N"])
        + alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q"), width="stretch")
    with st.expander("Recorded hourly conditions for this day"):
        tbl = dh.assign(Weather=dh.weathersit.map(WEATHER)).rename(columns={
            "hr": "Hour", "temp_c": "Temp (°C)", "hum_pct": "Humidity (%)", "wind_kmh": "Wind (km/h)",
            "cnt": "Rentals", "casual": "Casual", "registered": "Registered"})
        tbl["Predicted (GAM)"] = pr.count_gam.values
        st.dataframe(tbl[["Hour", "Weather", "Temp (°C)", "Humidity (%)", "Wind (km/h)", "Rentals", "Casual",
                          "Registered", "Predicted (GAM)"]].round(1), hide_index=True)

# ---- model detail for the selected hour
st.subheader("Inside the prediction for the selected hour")
left, right = st.columns(2)
probs = pd.DataFrame({"model": ["CART", "Random Forest", "Gradient Boosting", "AdaBoost", "Stacking"],
                      "P(high demand)": [out.p_cart, out.p_rf, out.p_gb, out.p_ada, out.p_stack]})
left.altair_chart(
    alt.Chart(probs, title="Each Unit 5 model's probability of high demand").mark_bar().encode(
        x=alt.X("P(high demand):Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("model:N", sort=None),
        color=alt.condition(alt.datum.model == "Stacking", alt.value("#2a6f97"), alt.value("#9bb8cc")),
    ) + alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q"),
    width="stretch",
)
left.caption("The stacking ensemble combines the four base models with a logistic regression; dashed line = 0.5.")

temps = np.arange(-5, 40.5, 1.0)
rows = pd.concat([core.make_row(d, hr, w, t, hum, wind, day, models["atemp_fit"]) for t in temps])
curve = pd.DataFrame({"Temperature (°C)": temps, "Predicted rentals": core.predict(models, rows).count_gam.values})
right.altair_chart(
    alt.Chart(curve, title="GAM: predicted rentals vs temperature (other inputs fixed)").mark_line().encode(
        x="Temperature (°C):Q", y="Predicted rentals:Q")
    + alt.Chart(pd.DataFrame({"t": [temp_c]})).mark_rule(color="#e07a5f").encode(x="t:Q"),
    width="stretch",
)
right.caption(captions.gam_curve(art["ref"]))
