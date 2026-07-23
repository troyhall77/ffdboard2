"""
streamlit_app.py

The dashboard. A live program: Streamlit runs this script to render the page and
RE-RUNS it on every interaction, which is why data loading is cached.

It only reads from data/. It never calls ESPN or Sleeper, and it does no modeling
of its own -- the charts are drawn from percentiles the simulator already saved.
Missing files (weekly projections, rosters, matchups, recap, sim) degrade to
gentle placeholders.

Run locally with:  streamlit run app/streamlit_app.py
"""

from pathlib import Path

import altair as alt
import polars as pl
import streamlit as st

DATA = Path(__file__).parent.parent / "data"

# Must match PCTS in scripts/simulate_projections.py
PCTS = list(range(5, 100, 5))
PCT_COLS = [f"p{p:02d}" for p in PCTS]

st.set_page_config(page_title="League Dashboard", layout="wide")


@st.cache_data(ttl=3600)
def load(name):
    """Read a parquet file from data/, or None if it doesn't exist yet.
    ttl=3600 refreshes hourly so the daily data update gets picked up."""
    path = DATA / f"{name}.parquet"
    return pl.read_parquet(path) if path.exists() else None


def render_league():
    st.header("Standings")

    standings = load("standings")
    if standings is None:
        st.info("No standings data yet -- run the league fetch first.")
        return

    table = (
        standings
        .with_columns(
            (pl.col("wins").cast(pl.Utf8) + "-"
             + pl.col("losses").cast(pl.Utf8) + "-"
             + pl.col("ties").cast(pl.Utf8)).alias("Record")
        )
        .sort(["standing", "points_for"], descending=[False, True])
        .select(
            pl.col("team_name").alias("Team"),
            pl.col("owner").alias("Owner"),
            pl.col("Record"),
            pl.col("points_for").alias("PF"),
            pl.col("points_against").alias("PA"),
        )
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("This week's matchups")
    matchups = load("matchups")
    if matchups is None:
        st.caption("Matchups will appear here once the season starts.")
    else:
        st.dataframe(matchups, use_container_width=True, hide_index=True)

    st.subheader("Weekly recap")
    recap = DATA / "recap.md"
    if recap.exists():
        # encoding="utf-8" so em-dashes render correctly on any platform.
        st.markdown(recap.read_text(encoding="utf-8"))
    else:
        st.caption("The weekly recap will be generated here during the season.")


def range_chart(frame, n):
    """Floor-to-ceiling bars for the top n players, with a tick at the median.

    This is the comparison view: two players can share a projection and have very
    different risk, and that shows up as bar width."""
    top = (
        frame.drop_nulls("sim_floor")
        .head(n)
        .select(
            pl.col("name").alias("Player"),
            pl.col("sim_floor").alias("Floor"),
            pl.col("sim_median").alias("Median"),
            pl.col("sim_ceiling").alias("Ceiling"),
        )
    )
    if not top.height:
        return None

    order = top["Player"].to_list()   # already sorted by projection
    pdf = top.to_pandas()

    base = alt.Chart(pdf).encode(
        y=alt.Y("Player:N", sort=order, title=None)
    )
    bars = base.mark_bar(size=9, opacity=0.45, color="#4C78A8").encode(
        x=alt.X("Floor:Q", title="Simulated season points (PPR)",
                scale=alt.Scale(zero=False)),
        x2="Ceiling:Q",
        tooltip=["Player", "Floor", "Median", "Ceiling"],
    )
    ticks = base.mark_tick(thickness=2, size=16, color="#2C3E50").encode(
        x="Median:Q", tooltip=["Player", "Median"]
    )
    return (bars + ticks).properties(height=max(220, 22 * top.height))


def density_chart(row):
    """Approximate distribution for one player, rebuilt from saved percentiles.

    Each adjacent percentile pair holds 5% of the outcomes, so a narrow gap means
    outcomes are piled up there (high likelihood) and a wide gap means they're
    spread thin. That gives us a density without re-simulating."""
    vals = [row[c] for c in PCT_COLS]
    step = (PCTS[1] - PCTS[0]) / 100.0        # 0.05 of the probability mass
    xs, ys = [], []
    for a, b in zip(vals[:-1], vals[1:]):
        width = max(b - a, 1e-6)
        xs += [a, b]
        ys += [step / width, step / width]
    pdf = pl.DataFrame({"points": xs, "likelihood": ys}).to_pandas()

    area = alt.Chart(pdf).mark_area(
        interpolate="step-after", opacity=0.55, color="#4C78A8"
    ).encode(
        x=alt.X("points:Q", title="Season points (PPR)", scale=alt.Scale(zero=False)),
        y=alt.Y("likelihood:Q", title="Relative likelihood", axis=None),
    )

    marks = pl.DataFrame({
        "x": [row["sim_floor"], row["sim_median"], row["sim_ceiling"]],
        "label": ["Floor (p10)", "Median", "Ceiling (p90)"],
    }).to_pandas()
    rules = alt.Chart(marks).mark_rule(strokeDash=[4, 3], color="#2C3E50").encode(
        x="x:Q", tooltip=["label", "x"]
    )
    return (area + rules).properties(height=260)


def render_projections():
    st.header("Player projections")

    view = st.radio("View", ["Season", "Weekly"], horizontal=True)
    df = load("projections_season" if view == "Season" else "projections_weekly")
    if df is None:
        st.info("Weekly projections will populate once the season starts.")
        return

    labels = {"pts_ppr": "PPR", "pts_half_ppr": "Half PPR", "pts_std": "Standard"}
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        scoring = st.selectbox("Scoring", list(labels), format_func=lambda k: labels[k])
    with c2:
        pos = st.selectbox("Position", ["All", "QB", "RB", "WR", "TE", "K", "DEF"])
    with c3:
        query = st.text_input("Search player")

    # The simulation is a SEASON model, so attach its range only in season view.
    sim = load("projections_sim") if view == "Season" else None
    if sim is not None:
        df = df.join(sim, on="player_id", how="left")

    filtered = df
    if pos != "All":
        filtered = filtered.filter(pl.col("position") == pos)
    if query:
        filtered = filtered.filter(
            pl.col("name").str.to_lowercase().str.contains(query.lower(), literal=True)
        )
    filtered = filtered.sort(scoring, descending=True, nulls_last=True)

    has_sim = sim is not None and "sim_floor" in filtered.columns
    has_pcts = has_sim and all(c in filtered.columns for c in PCT_COLS)

    cols = [
        pl.col("name").alias("Player"),
        pl.col("position").alias("Pos"),
        pl.col("team").alias("Team"),
        pl.col(scoring).alias("Proj"),
    ]
    if has_sim:
        cols += [
            pl.col("sim_floor").alias("Floor"),
            pl.col("sim_median").alias("Median"),
            pl.col("sim_ceiling").alias("Ceiling"),
        ]
    st.dataframe(filtered.select(cols), use_container_width=True, hide_index=True)

    if not has_sim:
        st.caption("A model range for weekly projections will slot in later.")
        return

    st.caption("Floor / Ceiling = 10th-90th percentile outcome range from the "
               "season simulation (PPR, v1). Reflects weekly and injury luck, "
               "not uncertainty in the projection itself.")

    st.divider()
    st.subheader("Outcome ranges")
    st.caption("Bar spans floor to ceiling; the tick marks the median. Wider bar "
               "= riskier player. Note the simulation is PPR-based, so these "
               "ranges stay on the PPR scale even if you switch scoring above.")

    n = st.slider("Players to compare", 5, 40, 15)
    chart = range_chart(filtered, n)
    if chart is None:
        st.info("No simulated players in this filter (K and DEF aren't modeled yet).")
    else:
        st.altair_chart(chart, use_container_width=True)

    if has_pcts:
        st.subheader("Single player distribution")
        named = filtered.drop_nulls("sim_floor")
        if named.height:
            who = st.selectbox("Player", named["name"].to_list())
            row = named.filter(pl.col("name") == who).row(0, named=True)
            st.altair_chart(density_chart(row), use_container_width=True)
            st.caption(
                f"{who}: floor {row['sim_floor']:.0f} - median "
                f"{row['sim_median']:.0f} - ceiling {row['sim_ceiling']:.0f} PPR points."
            )
    else:
        st.caption("Re-run the simulator to enable per-player distribution curves.")


st.title("League Dashboard")
league_tab, projections_tab = st.tabs(["League", "Projections"])
with league_tab:
    render_league()
with projections_tab:
    render_projections()