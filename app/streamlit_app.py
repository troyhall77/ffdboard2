"""
streamlit_app.py

The dashboard. A live program: Streamlit runs this script to render the page and
RE-RUNS it on every interaction, which is why data loading is cached.

It only reads from data/. It never calls ESPN or Sleeper. Missing files (weekly
projections, rosters, matchups, recap, sim) degrade to gentle placeholders.

Run locally with:  streamlit run app/streamlit_app.py
"""

from pathlib import Path

import polars as pl
import streamlit as st

DATA = Path(__file__).parent.parent / "data"

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

    out = df
    if pos != "All":
        out = out.filter(pl.col("position") == pos)
    if query:
        out = out.filter(
            pl.col("name").str.to_lowercase().str.contains(query.lower(), literal=True)
        )
    out = out.sort(scoring, descending=True, nulls_last=True)

    cols = [
        pl.col("name").alias("Player"),
        pl.col("position").alias("Pos"),
        pl.col("team").alias("Team"),
        pl.col(scoring).alias("Proj"),
    ]
    has_sim = sim is not None and "sim_floor" in out.columns
    if has_sim:
        cols += [
            pl.col("sim_floor").alias("Floor"),
            pl.col("sim_ceiling").alias("Ceiling"),
        ]
    out = out.select(cols)
    st.dataframe(out, use_container_width=True, hide_index=True)

    if has_sim:
        st.caption("Floor / Ceiling = 10th-90th percentile outcome range from the "
                   "season simulation (PPR, v1). Reflects weekly and injury luck, "
                   "not uncertainty in the projection itself.")
    else:
        st.caption("A model range for weekly projections will slot in later.")


st.title("League Dashboard")
league_tab, projections_tab = st.tabs(["League", "Projections"])
with league_tab:
    render_league()
with projections_tab:
    render_projections()