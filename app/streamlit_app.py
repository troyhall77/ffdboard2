"""
streamlit_app.py

The dashboard. Unlike the fetch scripts (which run once and write files), this
is a live program: when someone opens the page, Streamlit runs this script to
render it, and RE-RUNS the whole thing every time they click a control. That
rerun model is why data loading is cached below -- otherwise every click would
re-read the files from disk.

It only reads from data/. It never calls ESPN or Sleeper. If a file isn't there
yet (weekly projections, rosters, matchups, the recap), the app shows a gentle
placeholder instead of erroring -- so it works today in the offseason and fills
in on its own as those files come alive during the season.

Run locally with:  streamlit run app/streamlit_app.py
"""

from pathlib import Path

import polars as pl
import streamlit as st

# Anchor data/ to the repo, not the working directory -- same fix we made in
# the fetch scripts, for the same reason.
DATA = Path(__file__).parent.parent / "data"

st.set_page_config(page_title="League Dashboard", layout="wide")


@st.cache_data(ttl=3600)
def load(name):
    """Read a parquet file from data/, or return None if it doesn't exist yet.

    ttl=3600 means the cache refreshes hourly, so the app picks up the daily
    data refresh without needing a manual restart."""
    path = DATA / f"{name}.parquet"
    return pl.read_parquet(path) if path.exists() else None


def render_league():
    st.header("Standings")

    standings = load("standings")
    if standings is None:
        st.info("No standings data yet -- run the league fetch first.")
        return

    # Combine the W/L/T columns into one readable record, then pick and rename
    # the columns we want to show.
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
        st.markdown(recap.read_text(encoding="utf-8"), unsafe_allow_html=True)
    else:
        st.caption("The weekly recap will be generated here during the season.")


def render_projections():
    st.header("Player projections")

    view = st.radio("View", ["Season", "Weekly"], horizontal=True)
    df = load("projections_season" if view == "Season" else "projections_weekly")
    if df is None:
        st.info("Weekly projections will populate once the season starts.")
        return

    # Three controls side by side: scoring system, position filter, name search.
    labels = {"pts_ppr": "PPR", "pts_half_ppr": "Half PPR", "pts_std": "Standard"}
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        scoring = st.selectbox("Scoring", list(labels), format_func=lambda k: labels[k])
    with c2:
        pos = st.selectbox("Position", ["All", "QB", "RB", "WR", "TE", "K", "DEF"])
    with c3:
        query = st.text_input("Search player")

    out = df
    if pos != "All":
        out = out.filter(pl.col("position") == pos)
    if query:
        # literal=True treats the search text as plain text, so a stray
        # character can't be misread as a regex pattern and crash the filter.
        out = out.filter(
            pl.col("name").str.to_lowercase().str.contains(query.lower(), literal=True)
        )

    out = out.sort(scoring, descending=True, nulls_last=True).select(
        pl.col("name").alias("Player"),
        pl.col("position").alias("Pos"),
        pl.col("team").alias("Team"),
        pl.col(scoring).alias("Proj"),
    )
    st.dataframe(out, use_container_width=True, hide_index=True)
    st.caption(f"{out.height} players shown. Projections from Sleeper -- a "
               "league-model column will slot in next to these later.")


st.title("League Dashboard")
league_tab, projections_tab = st.tabs(["League", "Projections"])
with league_tab:
    render_league()
with projections_tab:
    render_projections()