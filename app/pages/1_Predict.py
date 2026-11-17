import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import ui  # noqa: E402
from app import core  # noqa: E402

ui.setup("Predict hourly demand")
st = ui.st
art = ui.get_artifacts()
models, day = art["models"], art["day"]
WEATHER = {1: "Clear / partly cloudy", 2: "Mist / cloudy", 3: "Light rain or snow", 4: "Heavy rain / storm"}

c1, c2, c3 = st.columns(3)
d = c1.date_input("Date", date(2012, 8, 15))
hr = c1.selectbox("Hour of day", list(range(24)), index=8, format_func=lambda h: f"{h:02d}:00")
w = c2.selectbox("Weather", list(WEATHER), format_func=WEATHER.get)
temp_c = c2.slider("Temperature (°C)", -10.0, 45.0, 24.0, 0.5)
hum = c3.slider("Humidity (%)", 0, 100, 60)
wind = c3.slider("Wind speed (km/h)", 0.0, 67.0, 12.0, 0.5)

row = core.make_row(d, hr, w, temp_c, hum, wind, day, models["atemp_fit"])
out = core.predict(models, row).iloc[0]
cal = core.calendar_fields(d, day)
if cal["out_of_range"]:
    st.info("The models were trained on 2011–2012 only. This date is treated as 2012 (the later, busier year).")

thresh = models["thresh"]
m1, m2, m3 = st.columns(3)
m1.metric("Predicted rentals this hour (GAM)", f"{out.count_gam:,.0f}")
m2.metric("High-demand probability (stacking)", f"{out.p_stack:.2f}",
          f"{'HIGH' if out.high else 'LOW'} (> {thresh:.0f} rides = high)", delta_color="off")
m3.metric("Day type", "Working day" if cal["workingday"] else ("Holiday" if cal["holiday"] else "Weekend"))

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
right.caption("The GAM is a sum of smooth effects; demand rises with temperature up to about 30 °C, then falls.")
