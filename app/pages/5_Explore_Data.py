import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ui  # noqa: E402
from app import core  # noqa: E402

ui.setup("Explore the data")
st = ui.st
art = ui.get_artifacts()
hour, thresh = art["hour"], art["models"]["thresh"]
SEASON = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}
DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

st.markdown("The raw hourly records behind every model (UCI dataset 275, 17,379 hours). Filter them and see the "
            "patterns that the units model: the hour × working-day effect, rider types, weather and growth.")

f1, f2, f3, f4 = st.columns(4)
years = f1.multiselect("Year", [2011, 2012], default=[2011, 2012], key="ex_year")
seasons = f2.multiselect("Season", list(SEASON), default=list(SEASON), format_func=SEASON.get, key="ex_season")
dtype = f3.radio("Day type", ["All", "Working days", "Non-working days"], key="ex_dtype")
weathers = f4.multiselect("Weather", list(core.WEATHER), default=list(core.WEATHER), format_func=core.WEATHER.get,
                          key="ex_weather")

h = hour.assign(year=hour.yr + 2011)
m = h.year.isin(years) & h.season.isin(seasons) & h.weathersit.isin(weathers)
if dtype != "All":
    m &= h.workingday == (1 if dtype == "Working days" else 0)
sel = h[m]

if sel.empty:
    st.warning("No hours match these filters. Widen the selection.")
    st.stop()

a, b, c, e = st.columns(4)
a.metric("Hours selected", f"{len(sel):,}")
b.metric("Average rentals per hour", f"{sel.cnt.mean():,.1f}")
c.metric(f"High-demand hours (> {thresh:.0f})", f"{(sel.cnt > thresh).mean():.0%}")
e.metric("Registered riders", f"{sel.registered.sum() / sel.cnt.sum():.0%}")

t1, t2, t3, t4 = st.tabs(["Hour × weekday", "Casual vs registered", "Weather and temperature", "Daily timeline"])

with t1:
    hm = sel.groupby(["weekday", "hr"]).cnt.mean().reset_index()
    hm["day"] = hm.weekday.map(dict(enumerate(DAYS)))
    st.altair_chart(alt.Chart(hm, title="Average rentals by hour and day of week").mark_rect().encode(
        x=alt.X("hr:O", title="hour of day"), y=alt.Y("day:N", sort=DAYS[1:] + DAYS[:1], title=None),
        color=alt.Color("cnt:Q", title="rentals/hour", scale=alt.Scale(scheme="blues")),
        tooltip=["day:N", "hr:O", alt.Tooltip("cnt:Q", format=",.0f", title="mean rentals")]), width="stretch")
    st.caption("Weekdays light up at 08:00 and 17:00–18:00 (commuting); Saturday and Sunday peak in the early "
               "afternoon. This is the hour × working-day interaction found in Unit 1 and the two day shapes of Unit 3.")

with t2:
    cr = sel.assign(kind=np.where(sel.workingday == 1, "Working days", "Non-working days")) \
        .groupby(["kind", "hr"])[["casual", "registered"]].mean().reset_index() \
        .melt(["kind", "hr"], var_name="rider", value_name="rentals")
    st.altair_chart(alt.Chart(cr).mark_area(opacity=0.8).encode(
        x=alt.X("hr:Q", title="hour of day"), y=alt.Y("rentals:Q", stack=True, title="mean rentals per hour"),
        color=alt.Color("rider:N", title=None, scale=alt.Scale(domain=["registered", "casual"],
                                                                range=["#2a6f97", "#e07a5f"])),
        column=alt.Column("kind:N", title=None),
        tooltip=["kind:N", "hr:Q", "rider:N", alt.Tooltip("rentals:Q", format=",.1f")]).properties(width=320))
    st.caption("Registered members make the commute peaks; casual riders are a weekend, afternoon crowd. These two "
               "columns add up to the target, which is why the models are not allowed to see them (leakage).")

with t3:
    tb = sel.assign(temp_bin=(np.floor(sel.temp * 41 / 3) * 3).astype(int), Weather=sel.weathersit.map(core.WEATHER))
    tw = tb.groupby(["Weather", "temp_bin"]).agg(rentals=("cnt", "mean"), hours=("cnt", "size")).reset_index()
    tw = tw[tw.hours >= 10]
    st.altair_chart(alt.Chart(tw, title="Average rentals by temperature (3 °C bins) and weather").mark_line(
        point=True).encode(
        x=alt.X("temp_bin:Q", title="temperature (°C)"), y=alt.Y("rentals:Q", title="mean rentals per hour"),
        color=alt.Color("Weather:N", sort=list(core.WEATHER.values()), title=None),
        tooltip=["Weather:N", "temp_bin:Q", alt.Tooltip("rentals:Q", format=",.0f"), "hours:Q"]), width="stretch")
    st.caption("Demand climbs with warmth until about 30 °C, as the Unit 5 GAM learned. Mist barely changes it; light "
               "rain or snow lowers it at most temperatures (the hot rainy bins hold few hours, often brief summer "
               "showers). Bins with fewer than 10 hours are hidden, so storms barely appear at all.")
    wt = sel.groupby("weathersit").agg(hours=("cnt", "size"), mean=("cnt", "mean"),
                                       high=("cnt", lambda s: (s > thresh).mean())).reset_index()
    wt["Weather"] = wt.weathersit.map(core.WEATHER)
    st.dataframe(wt[["Weather", "hours", "mean", "high"]].rename(columns={
        "hours": "Hours", "mean": "Mean rentals/hour", "high": "Share high-demand"}).round(3), hide_index=True)

with t4:
    dd = sel.groupby("date").cnt.sum().reset_index()
    dd["period"] = np.where(dd.date < pd.Timestamp("2012-08-07"), "training", "test")
    st.altair_chart(alt.Chart(dd, title="Rentals per day (selected hours only)").mark_bar(width=1).encode(
        x=alt.X("date:T", title=None), y=alt.Y("cnt:Q", title="rentals"),
        color=alt.Color("period:N", title=None, scale=alt.Scale(domain=["training", "test"],
                                                                 range=["#9bb8cc", "#2a6f97"])),
        tooltip=[alt.Tooltip("date:T", format="%a %d %b %Y"), alt.Tooltip("cnt:Q", format=",")]), width="stretch")
    yr = h.groupby("year").cnt.sum()
    st.caption(f"2012 had {yr[2012] / yr[2011] - 1:.0%} more rentals than 2011. The models trained up to "
               "7 Aug 2012 11:00 (light bars) and were tested on the rest (dark bars). Sharp dips are storms, "
               "for example Hurricane Sandy on 29–30 Oct 2012.")

with st.expander("Show the filtered hourly records"):
    cols = ["dteday", "hr", "season", "workingday", "weathersit", "temp", "hum", "windspeed", "casual", "registered",
            "cnt"]
    st.dataframe(sel[cols].head(2000), hide_index=True)
    st.download_button("Download filtered records (CSV)", sel[cols].to_csv(index=False).encode(),
                       "bike_hours_filtered.csv", "text/csv")
    st.caption("Weather columns are on the dataset's normalised scale (temp = °C / 41, hum = % / 100, "
               "windspeed = km/h / 67). The table shows up to 2,000 rows; the download has all of them.")
