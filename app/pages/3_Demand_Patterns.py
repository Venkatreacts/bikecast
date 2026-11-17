import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402

import ui  # noqa: E402

ui.setup("Demand patterns")
st = ui.st
art = ui.get_artifacts()
days, prof, hmm, ref = art["days"], art["kmeans_profiles"], art["hmm_test"], art["ref"]

# name clusters by the day type most of their members have
wd_share = days.groupby("cluster").workingday.mean()
names = {c: ("Commute shape (mostly working days)" if s > 0.5 else "Leisure shape (mostly weekends/holidays)")
         for c, s in wd_share.items()}
days["pattern"] = days.cluster.map(names)
prof["pattern"] = prof.cluster.map(names)

t1, t2, t3 = st.tabs(["K-means day shapes (Unit 3)", "PCA map of days (Unit 3)", "HMM demand states (Unit 4)"])

with t1:
    st.altair_chart(alt.Chart(prof, title="Average share of a day's rentals in each hour, per K-means cluster")
                    .mark_line(point=True).encode(x=alt.X("hr:Q", title="hour of day"),
                                                  y=alt.Y("share:Q", title="share of daily rentals"),
                                                  color=alt.Color("pattern:N", title=None)), width="stretch")
    k = ref["kmeans"]
    st.caption(f"Silhouette ({k['sil']:.2f}) and Davies–Bouldin ({k['DB']:.2f}) both choose K = 2. The clusters "
               f"match the working-day flag almost exactly (ARI {k['ARI_workingday']:.2f}): commuters create 8:00 "
               "and 17:00 peaks, leisure riders a midday hump. Three near-empty days (e.g. Hurricane Sandy) "
               "are excluded.")
    st.dataframe(pd.DataFrame(ref["kmeans_table"]).round(3), hide_index=True)

with t2:
    colour = st.radio("Colour days by", ["K-means cluster", "Working day flag"], horizontal=True)
    days["working day"] = days.workingday.map({1: "working day", 0: "non-working day"})
    field = "pattern" if colour == "K-means cluster" else "working day"
    st.altair_chart(alt.Chart(days, title="Each day's hourly profile projected onto the first two principal components")
                    .mark_circle(size=40, opacity=0.7).encode(
                        x=alt.X("pc1:Q", title="PC 1"), y=alt.Y("pc2:Q", title="PC 2"),
                        color=alt.Color(f"{field}:N", title=None),
                        tooltip=["date:T", "total:Q", "pattern:N"]), width="stretch")
    st.caption("Two well-separated groups: splitting on PC 1 alone reproduces the K-means clusters for 99.9 % of "
               "days. Only 11 days sit in the 'wrong' group for their calendar flag: mostly working days around "
               "Thanksgiving and Christmas that behave like leisure days, and 3 holidays that look like commute days.")

with t3:
    hmm["week"] = (hmm.date - pd.to_timedelta(hmm.date.dt.weekday, unit="D")).dt.date
    weeks = sorted(hmm.week.unique())
    default = weeks.index(pd.Timestamp("2012-10-15").date())
    wk = st.selectbox("Week starting (Monday)", weeks, index=default, format_func=lambda w: f"{w:%d %b %Y}")
    sel = hmm[hmm.week == wk].reset_index(drop=True)
    sel["hour index"] = range(len(sel))
    sel["state"] = "S" + sel.state.astype(str)
    base = alt.Chart(sel, title="Rentals per hour with the hidden state decoded by Viterbi")
    st.altair_chart(base.mark_line(color="#999999").encode(x="hour index:Q", y=alt.Y("cnt:Q", title="rentals"))
                    + base.mark_line(interpolate="step-after", color="#e07a5f").encode(x="hour index:Q", y="state_cnt:Q")
                    + base.mark_circle(size=25).encode(x="hour index:Q", y="cnt:Q",
                                                       color=alt.Color("state:N", title="state")),
                    width="stretch")
    st.caption("Orange steps = typical rentals of the decoded state (S0 lowest … S3 highest). States follow the "
               "daily rhythm: early-morning hours sit in S0, daytime and commute peaks mostly in S3.")
    A = pd.DataFrame(ref["hmm"]["A"], index=[f"from S{i}" for i in range(4)], columns=[f"to S{i}" for i in range(4)])
    st.markdown("**Learned transition matrix (Baum–Welch)**")
    st.dataframe(A)
    st.caption("High diagonal values mean states persist for several hours (mean dwell "
               + ", ".join(f"S{i} {d:.1f} h" for i, d in enumerate(ref["hmm"]["dwell"])) + ").")
