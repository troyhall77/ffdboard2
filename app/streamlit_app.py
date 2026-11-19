"""
streamlit_app.py

The dashboard. A live program: Streamlit runs this script to render the page and
RE-RUNS it on every interaction, which is why data loading is cached.

It only reads from data/. It never calls ESPN or Sleeper, and it does no modeling
of its own -- every range, win probability and playoff odd was computed by the
scripts and saved; distribution curves are rebuilt from saved percentiles.
Missing files degrade to gentle placeholders.

Run locally with:  streamlit run app/streamlit_app.py
"""

from pathlib import Path

import altair as alt
import polars as pl
import streamlit as st

DATA = Path(__file__).parent.parent / "data"

# Must match PCTS in scripts/weekly_model.py and scripts/simulate_projections.py
PCTS = list(range(5, 100, 5))
PCT_COLS = [f"p{p:02d}" for p in PCTS]

# Categorical palette, fixed order (validated for colorblind separation). A color
# always means the same thing: slot 1 = Sleeper, slot 2 = our model, etc.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
METHOD_COLORS = {"sleeper": PALETTE[0], "blend": PALETTE[1], "espn": PALETTE[2], "ppg": PALETTE[3]}
METHOD_LABELS = {"sleeper": "Sleeper", "blend": "Our model", "espn": "ESPN",
                 "ppg": "Season PPG"}
INK = "#52514e"

st.set_page_config(page_title="League Dashboard", layout="wide")


@st.cache_data(ttl=3600)
def load(name):
    """Read a parquet file from data/, or None if it doesn't exist yet.
    ttl=3600 refreshes hourly so the daily data update gets picked up."""
    path = DATA / f"{name}.parquet"
    return pl.read_parquet(path) if path.exists() else None


def pct(x):
    return f"{x:.0%}" if x is not None else "-"


# ======================================================================= League

def render_league():
    st.header("Standings")

    standings = load("standings")
    if standings is None:
        st.info("No standings data yet -- run the league fetch first.")
        return

    table = standings.with_columns(
        (pl.col("wins").cast(pl.Utf8) + "-" + pl.col("losses").cast(pl.Utf8) + "-"
         + pl.col("ties").cast(pl.Utf8)).alias("Record")
    )
    odds = load("playoff_odds")
    cols = [
        pl.col("team_name").alias("Team"), pl.col("owner").alias("Owner"),
        pl.col("Record"), pl.col("points_for").alias("PF"),
        pl.col("points_against").alias("PA"),
    ]
    config = {}
    if odds is not None:
        table = table.join(odds.drop("team_name"), on="team_id", how="left")
        cols += [
            pl.col("proj_wins").alias("Proj W"),
            (pl.col("playoff_pct") * 100).alias("Playoffs"),
            (pl.col("top_seed_pct") * 100).alias("#1 seed"),
            (pl.col("last_place_pct") * 100).alias("Last"),
        ]
        bar = dict(min_value=0, max_value=100, format="%.0f%%")
        config = {
            "Playoffs": st.column_config.ProgressColumn("Playoffs", **bar),
            "#1 seed": st.column_config.ProgressColumn("#1 seed", **bar),
            "Last": st.column_config.NumberColumn("Last place", format="%.0f%%"),
            "Proj W": st.column_config.NumberColumn("Proj W", format="%.1f"),
        }
    table = table.sort(["standing", "points_for"], descending=[False, True]).select(cols)
    st.dataframe(table, width="stretch", hide_index=True, column_config=config)
    if odds is not None:
        st.caption("Playoff odds simulate every remaining regular-season week 10,000 "
                   "times. Future weeks assume each team starts its best projected "
                   "lineup from its current roster; ties on wins break on points for.")

    render_matchups()

    st.subheader("Weekly recap")
    recap = DATA / "recap.md"
    if recap.exists():
        # encoding="utf-8" so em-dashes render correctly on any platform.
        st.markdown(recap.read_text(encoding="utf-8"))
    else:
        st.caption("The weekly recap will be generated here during the season.")


def win_prob_chart(odds, col):
    """One bar per game split at the home team's win probability. Home on the
    left (slot 1), away on the right (slot 2), 50% rule for reference."""
    rows = []
    for r in odds.iter_rows(named=True):
        label = f"{r['home_team']}  vs  {r['away_team']}"
        p = r[col]
        rows += [
            {"Game": label, "Team": r["home_team"], "Side": "Home", "start": 0.0,
             "end": p, "Win %": p},
            {"Game": label, "Team": r["away_team"], "Side": "Away", "start": p,
             "end": 1.0, "Win %": 1 - p},
        ]
    pdf = pl.DataFrame(rows).to_pandas()
    order = [f"{r['home_team']}  vs  {r['away_team']}" for r in odds.iter_rows(named=True)]
    base = alt.Chart(pdf).encode(y=alt.Y("Game:N", sort=order, title=None,
                                         axis=alt.Axis(labelLimit=420)))
    bars = base.mark_bar(size=18, cornerRadius=4, stroke="white", strokeWidth=2).encode(
        x=alt.X("start:Q", title="Win probability", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        x2="end:Q",
        color=alt.Color("Side:N", scale=alt.Scale(domain=["Home", "Away"],
                                                  range=PALETTE[:2]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Team", alt.Tooltip("Win %:Q", format=".0%")],
    )
    mid = alt.Chart(pl.DataFrame({"x": [0.5]}).to_pandas()).mark_rule(
        color=INK, strokeDash=[3, 3], opacity=0.6).encode(x="x:Q")
    labels = base.transform_filter(alt.datum["Win %"] >= 0.12).mark_text(
        color="white", fontWeight="bold").encode(
        x=alt.X("mid:Q"), text=alt.Text("Win %:Q", format=".0%"),
    ).transform_calculate(mid="(datum.start + datum.end) / 2")
    return (bars + mid + labels).properties(height=46 * len(order))


def render_matchups():
    st.subheader("This week's matchups")
    odds = load("matchup_odds")
    matchups = load("matchups")
    if odds is None or not odds.height:
        if matchups is None:
            st.caption("Matchups will appear here once the season starts.")
        else:
            st.dataframe(matchups, width="stretch", hide_index=True)
        return

    view = st.radio(
        "Lineups", ["As set right now", "If both start their best lineup"],
        horizontal=True,
        help="'As set' uses each manager's current starters (live -- final games "
             "count their real points; a starter on bye scores 0). 'Best lineup' "
             "assumes everyone fixes their lineup before kickoff.",
    )
    best = view.startswith("If")
    prob_col = "home_win_prob_best" if best else "home_win_prob"
    st.altair_chart(win_prob_chart(odds, prob_col), width="stretch")

    hm, am = ("home_best_mean", "away_best_mean") if best else ("home_mean", "away_mean")
    table = odds.select(
        pl.col("home_team").alias("Home"),
        pl.col("home_score").alias("Score"),
        pl.col(hm).round(1).alias("Proj"),
        (pl.col(prob_col) * 100).round(0).alias("Win %"),
        (100 - pl.col(prob_col) * 100).round(0).alias("Win % "),
        pl.col(am).round(1).alias("Proj "),
        pl.col("away_score").alias("Score "),
        pl.col("away_team").alias("Away"),
    )
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(f"Week {odds['week'][0]}. Each game simulated 10,000 times from player-"
               "level outcome distributions (see Model accuracy). Proj = average "
               "simulated final score.")


# ======================================================================= Projections

def _density_points(row):
    """Percentiles -> (points, likelihood) step coordinates for one player.

    Each adjacent percentile pair holds 5% of the outcomes, so a narrow gap means
    outcomes are piled up there (high likelihood) and a wide gap means they're
    spread thin. That gives us a density without re-simulating."""
    vals = [row[c] for c in PCT_COLS]
    step = (PCTS[1] - PCTS[0]) / 100.0
    xs, ys = [], []
    for a, b in zip(vals[:-1], vals[1:]):
        width = max(b - a, 1e-6)
        xs += [a, b]
        ys += [step / width, step / width]
    return xs, ys


def density_chart(row, x_title):
    xs, ys = _density_points(row)
    pdf = pl.DataFrame({"points": xs, "likelihood": ys}).to_pandas()
    area = alt.Chart(pdf).mark_area(
        interpolate="step-after", opacity=0.55, color=PALETTE[0]
    ).encode(
        x=alt.X("points:Q", title=x_title, scale=alt.Scale(zero=False)),
        y=alt.Y("likelihood:Q", title="Relative likelihood", axis=None),
        tooltip=[alt.Tooltip("points:Q", title="points", format=".1f")],
    )
    marks = pl.DataFrame({
        "x": [row["sim_floor"], row["sim_median"], row["sim_ceiling"]],
        "label": ["Floor (p10)", "Median", "Ceiling (p90)"],
    }).to_pandas()
    rules = alt.Chart(marks).mark_rule(strokeDash=[4, 3], color=INK, strokeWidth=2).encode(
        x="x:Q", tooltip=["label", alt.Tooltip("x:Q", format=".1f")]
    )
    return (area + rules).properties(height=300)


def compare_density_chart(rows, x_title):
    """Overlay several players' distributions. Lines rather than filled areas so
    up to five overlapping curves stay readable."""
    frames = []
    for row in rows:
        xs, ys = _density_points(row)
        frames.append(pl.DataFrame({
            "points": xs, "likelihood": ys, "Player": [row["name"]] * len(xs),
        }))
    pdf = pl.concat(frames).to_pandas()
    names = [r["name"] for r in rows]
    return alt.Chart(pdf).mark_line(
        interpolate="step-after", strokeWidth=2, opacity=0.9
    ).encode(
        x=alt.X("points:Q", title=x_title, scale=alt.Scale(zero=False)),
        y=alt.Y("likelihood:Q", title="Relative likelihood", axis=None),
        color=alt.Color("Player:N", title=None, sort=names,
                        scale=alt.Scale(domain=names, range=PALETTE[:len(names)]),
                        legend=alt.Legend(orient="top")),
        tooltip=["Player", alt.Tooltip("points:Q", title="points", format=".1f")],
    ).properties(height=320)


def compare_range_chart(rows, x_title):
    """Floor-to-ceiling bars for the selected players, tick at the median.
    Wider bar = riskier player."""
    names = [r["name"] for r in rows]
    pdf = pl.DataFrame({
        "Player": names,
        "Floor": [r["sim_floor"] for r in rows],
        "Median": [r["sim_median"] for r in rows],
        "Ceiling": [r["sim_ceiling"] for r in rows],
    }).to_pandas()
    base = alt.Chart(pdf).encode(y=alt.Y("Player:N", sort=names, title=None))
    bars = base.mark_bar(size=11, opacity=0.6, cornerRadius=4).encode(
        x=alt.X("Floor:Q", title=x_title, scale=alt.Scale(zero=False)),
        x2="Ceiling:Q",
        color=alt.Color("Player:N", legend=None,
                        scale=alt.Scale(domain=names, range=PALETTE[:len(names)])),
        tooltip=["Player", "Floor", "Median", "Ceiling"],
    )
    ticks = base.mark_tick(thickness=2, size=18, color=INK).encode(
        x="Median:Q", tooltip=["Player", "Median"]
    )
    return (bars + ticks).properties(height=max(120, 34 * len(rows)))


VIEWS = {
    "This week": {
        "file": "projections_week_sim", "x_title": "Points this week (PPR)",
        "caption": "Proj = Sleeper's projection nudged toward season production "
                   "(our model). Floor / Median / Ceiling = 10th / 50th / 90th "
                   "percentile of this week's outcome, from how players at this "
                   "position and projection level actually scored vs. projection "
                   "this season. Includes the chance he's a late scratch.",
    },
    "Rest of season": {
        "file": "projections_ros", "x_title": "Final season total (PPR)",
        "caption": "Season total = points already scored + every remaining week "
                   "simulated (byes count as zero; injuries last multiple weeks). "
                   "ROS = projected points still to come.",
    },
    "Preseason": {
        "file": "projections_season", "x_title": "Season points (PPR)",
        "caption": "Sleeper's preseason season totals with the original v1 range "
                   "(weekly + injury luck around a fixed projection). Kept for "
                   "reference; it doesn't know anything that's happened this year.",
    },
}


def projection_table(view, df, scoring):
    if view == "This week":
        cols = [pl.col("sleeper_proj").alias("Sleeper"), pl.col("model_proj").alias("Proj")]
    elif view == "Rest of season":
        cols = [pl.col("ytd_pts").alias("So far"), pl.col("ytd_games").alias("GP"),
                pl.col("ros_mean").alias("ROS")]
    else:
        cols = [pl.col(scoring).alias("Proj")]
    if "sim_floor" in df.columns:
        cols += [pl.col("sim_floor").alias("Floor"), pl.col("sim_median").alias("Median"),
                 pl.col("sim_ceiling").alias("Ceiling")]
    return df.select(pl.col("name").alias("Player"), pl.col("position").alias("Pos"),
                     pl.col("team").alias("Team"), *cols)


def render_projections():
    st.header("Player projections")

    available = [v for v, spec in VIEWS.items() if load(spec["file"]) is not None]
    if not available:
        st.info("No projections yet -- run the projection scripts first.")
        return
    view = st.radio("View", available, horizontal=True)
    spec = VIEWS[view]
    df = load(spec["file"])

    labels = {"pts_ppr": "PPR", "pts_half_ppr": "Half PPR", "pts_std": "Standard"}
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if view == "Preseason":
            scoring = st.selectbox("Scoring", list(labels), format_func=lambda k: labels[k])
        else:
            scoring = "pts_ppr"
            st.selectbox("Scoring", ["PPR (league)"], disabled=True,
                         help="In-season models are fit to this league's scoring (PPR).")
    with c2:
        pos = st.selectbox("Position", ["All", "QB", "RB", "WR", "TE", "K", "DEF"])
    with c3:
        query = st.text_input("Search player")

    if view == "Preseason":
        sim = load("projections_sim")
        if sim is not None:
            df = df.join(sim, on="player_id", how="left")
        sort_col = scoring
    else:
        sort_col = "sim_median"

    filtered = df
    if pos != "All":
        filtered = filtered.filter(pl.col("position") == pos)
    if query:
        filtered = filtered.filter(
            pl.col("name").str.to_lowercase().str.contains(query.lower(), literal=True))
    filtered = filtered.sort(sort_col, descending=True, nulls_last=True)

    st.dataframe(projection_table(view, filtered, scoring),
                 width="stretch", hide_index=True)
    st.caption(spec["caption"])

    if not all(c in filtered.columns for c in PCT_COLS):
        return
    named = filtered.drop_nulls("sim_floor")
    if not named.height:
        st.info("No simulated players in this filter.")
        return

    st.divider()
    st.subheader("Outcome distribution")
    who = st.selectbox("Player", named["name"].to_list())
    row = named.filter(pl.col("name") == who).row(0, named=True)
    st.altair_chart(density_chart(row, spec["x_title"]), width="stretch")
    st.caption(f"{who}: floor {row['sim_floor']:.1f} - median {row['sim_median']:.1f} - "
               f"ceiling {row['sim_ceiling']:.1f} PPR points.")

    st.divider()
    st.subheader("Compare players")
    st.caption("Type to search, click to add. Curves further right are better; "
               "flatter and wider means riskier.")
    names = named["name"].to_list()
    picks = st.multiselect("Players to compare", names, default=names[:2], max_selections=5)
    if len(picks) < 2:
        st.caption("Pick at least two players to compare.")
        return
    rows = [named.filter(pl.col("name") == p).row(0, named=True) for p in picks]
    st.altair_chart(compare_density_chart(rows, spec["x_title"]), width="stretch")
    st.altair_chart(compare_range_chart(rows, spec["x_title"]), width="stretch")
    st.caption("Bars span floor (p10) to ceiling (p90); the tick marks the median.")


# ======================================================================= Model accuracy

def render_accuracy():
    st.header("How good are these projections?")
    bt = load("backtest")
    if bt is None:
        st.info("Run scripts/backtest.py to score the projections.")
        return

    test_weeks = bt["test_weeks"][0]
    st.markdown(
        f"Every projection below was made **before** its week was played, then "
        f"compared with what actually happened. Our model's settings were fit on "
        f"earlier weeks and scored on **weeks {test_weeks}**, which it never saw. "
        "Lower error is better."
    )

    # ---- headline tiles
    allp = bt.filter((pl.col("subset") == "All projected players") & (pl.col("position") == "ALL"))
    mae = {r["method"]: r["mae"] for r in allp.iter_rows(named=True)}
    cal = load("backtest_calibration")
    cov = (cal.filter(pl.col("position") == "ALL")["coverage"][0]
           if cal is not None else None)
    t1, t2, t3 = st.columns(3)
    t1.metric("Sleeper avg miss", f"{mae.get('sleeper', 0):.2f} pts")
    t2.metric("Our model avg miss", f"{mae.get('blend', 0):.2f} pts",
              delta=f"{mae.get('blend', 0) - mae.get('sleeper', 0):+.2f} vs Sleeper",
              delta_color="inverse")
    t3.metric("80% ranges that held", pct(cov), help="Target: 80%. Lower would "
              "mean the floors and ceilings are too narrow.")

    # ---- MAE by position
    st.subheader("Average miss by position")
    subset = st.radio("Players", bt["subset"].unique(maintain_order=True).to_list(),
                      horizontal=True,
                      help="ESPN only gives us projections for rostered players, so "
                           "it's compared on that subset.")
    sub = bt.filter((pl.col("subset") == subset) & (pl.col("position") != "ALL"))
    methods = [m for m in METHOD_LABELS if m in sub["method"].unique().to_list()]
    pdf = sub.with_columns(
        pl.col("method").replace(METHOD_LABELS).alias("Method")
    ).to_pandas()
    method_names = [METHOD_LABELS[m] for m in methods]
    chart = alt.Chart(pdf).mark_bar(cornerRadiusEnd=4, stroke="white", strokeWidth=2).encode(
        x=alt.X("mae:Q", title="Mean absolute error (PPR points)"),
        y=alt.Y("Method:N", sort=method_names, title=None, axis=None),
        color=alt.Color("Method:N", sort=method_names, title=None,
                        scale=alt.Scale(domain=method_names,
                                        range=[METHOD_COLORS[m] for m in methods]),
                        legend=alt.Legend(orient="top")),
        row=alt.Row("position:N", title=None,
                    sort=["QB", "RB", "WR", "TE", "K", "DEF"],
                    header=alt.Header(labelAngle=0, labelAlign="left")),
        tooltip=["position", "Method", alt.Tooltip("mae:Q", format=".2f", title="MAE"),
                 alt.Tooltip("bias:Q", format="+.2f", title="Bias (actual - proj)"),
                 alt.Tooltip("n:Q", title="Games")],
    ).properties(height=18 * len(methods))
    st.altair_chart(chart, width="stretch")
    with st.expander("Table view"):
        st.dataframe(sub.pivot(on="method", index="position", values="mae")
                     .rename(METHOD_LABELS, strict=False),
                     width="stretch", hide_index=True)

    # ---- by week
    trend = load("backtest_weekly")
    if trend is not None:
        st.subheader("Average miss by week")
        tdf = trend.with_columns(
            pl.col("method").replace(METHOD_LABELS).alias("Method")).to_pandas()
        names = [METHOD_LABELS["sleeper"], METHOD_LABELS["blend"]]
        line = alt.Chart(tdf).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64)).encode(
            x=alt.X("week:O", title="Week", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("mae:Q", title="MAE (PPR points)", scale=alt.Scale(zero=False)),
            color=alt.Color("Method:N", sort=names, title=None,
                            scale=alt.Scale(domain=names,
                                            range=[METHOD_COLORS["sleeper"], METHOD_COLORS["blend"]]),
                            legend=alt.Legend(orient="top")),
            tooltip=["week", "Method", alt.Tooltip("mae:Q", format=".2f"), "n"],
        )
        first_holdout = int(trend.filter(pl.col("holdout"))["week"].min())
        rule = alt.Chart(pl.DataFrame({"week": [first_holdout]}).to_pandas()).mark_rule(
            color=INK, strokeDash=[4, 3]).encode(x="week:O")
        st.altair_chart((line + rule).properties(height=280), width="stretch")
        st.caption(f"Dashed line = start of the holdout weeks. Weeks before it were "
                   "used to fit the model, so the model's error there is flattered.")

    # ---- calibration
    if cal is not None:
        st.subheader("Do the floors and ceilings hold?")
        cdf = cal.filter(pl.col("position") != "ALL").to_pandas()
        bars = alt.Chart(cdf).mark_bar(cornerRadiusEnd=4, color=PALETTE[0], size=22).encode(
            x=alt.X("position:N", sort=["QB", "RB", "WR", "TE", "K", "DEF"], title=None,
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("coverage:Q", title="Actual results inside the p10-p90 range",
                    axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            tooltip=["position", alt.Tooltip("coverage:Q", format=".0%"),
                     alt.Tooltip("below_floor:Q", format=".0%", title="Below floor"),
                     alt.Tooltip("above_ceiling:Q", format=".0%", title="Above ceiling"),
                     alt.Tooltip("n:Q", title="Games")],
        )
        target = alt.Chart(pl.DataFrame({"y": [0.8]}).to_pandas()).mark_rule(
            color=INK, strokeDash=[4, 3], strokeWidth=2).encode(y="y:Q")
        st.altair_chart((bars + target).properties(height=260), width="stretch")
        st.caption("Dashed line = the 80% target. Well calibrated means 10% of results "
                   "land below the floor and 10% above the ceiling.")

    with st.expander("How the model works"):
        st.markdown(
            "- **Mean:** Sleeper's weekly projection, nudged toward the player's "
            "season points-per-game by an amount that grows with games played. How "
            "fast it trusts production is fit per position, and the backtest's "
            "verdict is that Sleeper already bakes recent production in, so the "
            "nudge is small.\n"
            "- **Range:** not a textbook bell curve. For each position and "
            "projection level we store how actual scores compared with projection "
            "this season (the real mix of duds and blowups) and draw from that.\n"
            "- **Availability:** a measured late-scratch rate for this week; for "
            "future weeks, injuries that last ~3 weeks at 2025's games-played rate.\n"
            "- **Matchups and playoffs:** add up the starters' simulated weeks, "
            "10,000 times."
        )
        params = load("weekly_params")
        if params is not None:
            st.dataframe(
                params.select(
                    "position", pl.col("proj_floor").alias("tier from (pts)"), "n",
                    pl.col("ratio_q").list.get(2).round(2).alias("p10 x proj"),
                    pl.col("ratio_q").list.get(10).round(2).alias("median x proj"),
                    pl.col("ratio_q").list.get(18).round(2).alias("p90 x proj"),
                    pl.col("dnp_rate").round(3).alias("late scratch rate"),
                ),
                width="stretch", hide_index=True,
            )


st.title("League Dashboard")
league_tab, projections_tab, accuracy_tab = st.tabs(["League", "Projections", "Model accuracy"])
with league_tab:
    render_league()
with projections_tab:
    render_projections()
with accuracy_tab:
    render_accuracy()
