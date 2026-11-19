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

import html
from datetime import datetime, timezone
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

st.set_page_config(page_title="Gun Squad League Dashboard", layout="wide")


@st.cache_data(ttl=3600)
def load(name):
    """Read a parquet file from data/, or None if it doesn't exist yet.
    ttl=3600 refreshes hourly so the daily data update gets picked up."""
    path = DATA / f"{name}.parquet"
    return pl.read_parquet(path) if path.exists() else None


def pct(x):
    return f"{x:.0%}" if x is not None else "-"


def chart_surface():
    """Background behind Altair charts in the active theme. Bar strokes use it so
    the 2px gap between adjacent fills reads as a gap, not a white outline."""
    return "#0e1117" if getattr(st.context.theme, "type", None) == "dark" else "#ffffff"


# ======================================================================= Look & feel

def apply_theme():
    """Football-field backdrop with the content on a solid card above it.

    The field is pure CSS (no image to host): mowed turf stripes, a yard line
    every 10%, two rows of hash marks, and a vignette. Readability comes from the
    card -- 96% opaque in the active theme's surface color -- so text and charts
    sit on the same background they always did; the field only shows around the
    edges and behind the header."""
    dark = getattr(st.context.theme, "type", None) == "dark"
    turf, card = ("#1d4a24", "rgba(20, 22, 20, 0.96)") if dark else \
                 ("#2f7a3a", "rgba(252, 252, 251, 0.96)")
    st.markdown(f"""
<style>
.stApp {{
  background-color: {turf};
  background-image:
    radial-gradient(ellipse at center, transparent 45%, rgba(0,0,0,0.38) 100%),
    repeating-linear-gradient(90deg, transparent 0 calc(10% - 2px),
                              rgba(255,255,255,0.30) calc(10% - 2px) 10%),
    repeating-linear-gradient(90deg, transparent 0 calc(2% - 1px),
                              rgba(255,255,255,0.30) calc(2% - 1px) 2%),
    repeating-linear-gradient(90deg, transparent 0 calc(2% - 1px),
                              rgba(255,255,255,0.30) calc(2% - 1px) 2%),
    repeating-linear-gradient(90deg, rgba(255,255,255,0.045) 0 5%,
                              rgba(0,0,0,0.045) 5% 10%);
  background-size: 100% 100%, 100% 100%, 100% 16px, 100% 16px, 100% 100%;
  background-position: 0 0, 0 0, 0 36%, 0 64%, 0 0;
  background-repeat: no-repeat, no-repeat, repeat-x, repeat-x, no-repeat;
  background-attachment: fixed;
}}
header[data-testid="stHeader"] {{ background: transparent; color: #fff; }}
header[data-testid="stHeader"] button, header[data-testid="stHeader"] a {{ color: #fff !important; }}
/* The card is inset so the field frames it on every side. */
.stMainBlockContainer, .block-container {{
  background: {card};
  border-radius: 18px;
  box-shadow: 0 10px 40px rgba(0,0,0,0.35);
  width: min(1440px, calc(100% - 7rem));
  max-width: none;
  margin: 3.25rem auto 3rem;
  padding: 1.75rem 2.25rem 2.5rem;
}}
@media (max-width: 640px) {{
  .stMainBlockContainer, .block-container {{
    width: calc(100% - 1.25rem);
    padding: 1.25rem 1rem 2rem; border-radius: 12px; margin-top: 2.75rem;
  }}
}}

/* ---- League wire ticker ---- */
.wire {{
  display: flex; align-items: stretch; overflow: hidden;
  border-radius: 10px; background: #111312; color: #f0efec;
  border: 1px solid rgba(255,255,255,0.12);
  font-size: 0.92rem; margin: 0 0 1.1rem;
}}
.wire-label {{
  flex: none; padding: 0.55rem 0.85rem; background: #eb6834; color: #111312;
  font-weight: 800; letter-spacing: 0.06em; font-size: 0.78rem;
  display: flex; align-items: center;
}}
.wire-window {{ flex: 1; overflow: hidden; position: relative; }}
.wire-track {{
  display: inline-flex; white-space: nowrap; gap: 2.5rem;
  padding: 0.55rem 0 0.55rem 1.25rem;
  animation: wire-scroll 45s linear infinite;
}}
.wire:hover .wire-track, .wire:focus-within .wire-track {{ animation-play-state: paused; }}
@keyframes wire-scroll {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
@media (prefers-reduced-motion: reduce) {{
  .wire-track {{ animation: none; white-space: normal; flex-wrap: wrap; row-gap: 0.4rem; }}
  .wire-track [aria-hidden="true"] {{ display: none; }}
}}
.wire-item {{ display: inline-flex; align-items: center; gap: 0.5rem; }}
.wire-when {{ color: #a9a89f; font-size: 0.82rem; }}
.wire-link {{
  flex: none; display: flex; align-items: center; padding: 0 0.95rem;
  color: #f0efec !important; text-decoration: none !important; font-weight: 600;
  border-left: 1px solid rgba(255,255,255,0.14);
}}
.wire-link:hover {{ background: rgba(255,255,255,0.08); }}
.pill {{
  display: inline-block; padding: 0.05rem 0.45rem; border-radius: 999px;
  font-size: 0.7rem; font-weight: 800; letter-spacing: 0.05em; color: #111312;
}}
.pill-add {{ background: #6fd3a8; }}
.pill-waiver {{ background: #8fbaf0; }}
.pill-drop {{ background: #c9c8c0; }}
.pill-trade {{ background: #f6c453; }}

/* ---- Feed ---- */
.feed-day {{ margin: 1.4rem 0 0.4rem; font-weight: 700; font-size: 0.95rem; opacity: 0.75; }}
.feed-row {{
  display: flex; gap: 0.75rem; align-items: baseline; padding: 0.55rem 0;
  border-bottom: 1px solid rgba(128,128,128,0.18);
}}
.feed-time {{ flex: none; width: 4.75rem; font-size: 0.85rem; opacity: 0.65; }}
.feed-pill {{ flex: none; width: 4.5rem; }}
</style>
""", unsafe_allow_html=True)


# ======================================================================= League wire

KIND_PILL = {"Add": "add", "Waiver": "waiver", "Drop": "drop", "Trade": "trade"}


def _names(players):
    return ", ".join(f"<b>{html.escape(p)}</b>" for p in players)


@st.cache_data(ttl=3600)
def league_moves():
    """Collapse the transaction log (one row per player moved) into one entry
    per transaction, each with a kind and a sentence of HTML."""
    tx = load("transactions")
    if tx is None or "move_id" not in tx.columns:
        return []
    tx = tx.with_columns(
        when=pl.from_epoch("date", time_unit="ms").dt.replace_time_zone("UTC")
        .dt.convert_time_zone("America/New_York"))
    moves = []
    for (move_id,), g in tx.group_by(["move_id"], maintain_order=True):
        rows = g.to_dicts()
        when = rows[0]["when"]
        teams = sorted({r["team"] for r in rows} | {r["to_team"] for r in rows if r["to_team"]})
        if any(r["action"] == "TRADED" for r in rows):
            gets = {}
            for r in rows:
                gets.setdefault(r["to_team"], []).append(r["player"])
            sides = [f"{html.escape(t)} get {_names(p)}" for t, p in gets.items()]
            text = "Trade: " + " · ".join(sides)
            kind = "Trade"
        else:
            team = html.escape(rows[0]["team"])
            claimed = [r["player"] for r in rows if r["action"] == "WAIVER ADDED"]
            added = [r["player"] for r in rows if r["action"] == "FA ADDED"]
            dropped = [r["player"] for r in rows if r["action"] == "DROPPED"]
            parts = []
            if claimed:
                parts.append(f"claimed {_names(claimed)} off waivers")
            if added:
                parts.append(f"added {_names(added)}")
            if dropped:
                parts.append(f"dropped {_names(dropped)}")
            text = f"{team} " + ", ".join(parts)
            kind = "Waiver" if claimed else "Add" if added else "Drop"
        moves.append({"id": move_id, "when": when, "kind": kind, "teams": teams,
                      "players": [r["player"] for r in rows], "html": text})
    moves.sort(key=lambda m: m["when"], reverse=True)
    return moves


def ago(when):
    secs = (datetime.now(timezone.utc) - when.astimezone(timezone.utc)).total_seconds()
    if secs < 3600:
        return f"{max(1, int(secs // 60))}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    days = int(secs // 86400)
    return f"{days}d ago" if days < 14 else when.strftime("%b %d")


def pill(kind):
    return f'<span class="pill pill-{KIND_PILL[kind]}">{kind.upper()}</span>'


def render_ticker():
    """The five most recent moves, scrolling. Hover (or focus) pauses it;
    reduced-motion users get a static wrapped list instead."""
    moves = league_moves()[:5]
    if not moves:
        return
    items = "".join(
        f'<span class="wire-item">{pill(m["kind"])} {m["html"]} '
        f'<span class="wire-when">{ago(m["when"])}</span></span>'
        for m in moves
    )
    # The track holds two copies so the loop is seamless; screen readers and
    # reduced-motion users get just the first.
    st.markdown(
        f'<div class="wire" role="region" aria-label="Latest league moves">'
        f'<div class="wire-label">LEAGUE WIRE</div>'
        f'<div class="wire-window"><div class="wire-track">'
        f'{items}<span class="wire-item" aria-hidden="true" style="display:contents">{items}</span>'
        f'</div></div>'
        f'<a class="wire-link" href="?view=moves" target="_self">All moves &rarr;</a>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_moves_feed():
    """Every transaction this season, newest first, with filters."""
    st.markdown('<a href="./" target="_self">&larr; Back to the dashboard</a>',
                unsafe_allow_html=True)
    st.title("League wire")
    moves = league_moves()
    if not moves:
        st.info("No league moves yet.")
        return

    all_teams = sorted({t for m in moves for t in m["teams"]})
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        teams = st.multiselect("Team", all_teams, placeholder="All teams")
    with c2:
        kinds = st.multiselect("Type", list(KIND_PILL), placeholder="All moves")
    with c3:
        query = st.text_input("Search player")

    shown = [
        m for m in moves
        if (not teams or set(teams) & set(m["teams"]))
        and (not kinds or m["kind"] in kinds)
        and (not query or any(query.lower() in p.lower() for p in m["players"]))
    ]
    counts = {k: sum(m["kind"] == k for m in shown) for k in KIND_PILL}
    st.caption(f"{len(shown)} of {len(moves)} moves · "
               + " · ".join(f"{v} {k.lower()}{'s' if v != 1 else ''}" for k, v in counts.items()))

    out, day = [], None
    for m in shown:
        w = m["when"]
        d = f"{w:%A, %B} {w.day}"   # no %-d: it isn't portable to Windows
        if d != day:
            out.append(f'<div class="feed-day">{d}</div>')
            day = d
        out.append(
            f'<div class="feed-row"><span class="feed-time">'
            f'{w.hour % 12 or 12}:{w:%M} {"am" if w.hour < 12 else "pm"}</span>'
            f'<span class="feed-pill">{pill(m["kind"])}</span><span>{m["html"]}</span></div>')
    st.markdown("".join(out) or "_Nothing matches those filters._", unsafe_allow_html=True)

    # Who's been busiest? Single series, so no legend -- the title names it.
    st.divider()
    st.subheader("Moves by team")
    by_team = (pl.DataFrame([{"Team": t, "kind": m["kind"]} for m in moves for t in m["teams"]])
               .group_by("Team").agg(Moves=pl.len()).sort("Moves", descending=True))
    chart = alt.Chart(by_team.to_pandas()).mark_bar(
        color=PALETTE[0], cornerRadiusEnd=4, size=16).encode(
        x=alt.X("Moves:Q", title="Transactions this season"),
        y=alt.Y("Team:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
        tooltip=["Team", "Moves"],
    ).properties(height=30 * by_team.height)
    st.altair_chart(chart, width="stretch")


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
    # Tall enough for every team -- no scrolling inside the standings.
    st.dataframe(table, width="stretch", hide_index=True, column_config=config,
                 height=35 * (table.height + 1) + 3)
    if odds is not None:
        st.caption("Playoff odds simulate every remaining regular-season week 10,000 "
                   "times. Future weeks assume each team starts its best projected "
                   "lineup from its current roster; ties on wins break on points for.")

    render_matchups()
    render_awards()


def weekly_awards():
    """Last completed week's five awards, as (emoji, title, winner, detail)
    tuples. Scores come from matchups_last; projections and bench points from
    the ESPN box scores (league_weekly) -- ESPN's pregame numbers, i.e. what
    managers saw when they set lineups. Returns (week, awards) or (None, [])."""
    last = load("matchups_last")
    lw = load("league_weekly")
    if last is None or not last.height or lw is None:
        return None, []
    week = int(last["week"][0])
    awards = []

    # Head-to-head margins (byes have no opponent and are skipped).
    games = []
    for g in last.drop_nulls(["home_team", "away_team"]).iter_rows(named=True):
        (w, ws), (l, ls) = sorted(
            [(g["home_team"].strip(), g["home_score"]), (g["away_team"].strip(), g["away_score"])],
            key=lambda t: t[1], reverse=True)
        if ws > ls:
            games.append((ws - ls, w, ws, l, ls))
    if games:
        m, w, ws, l, ls = max(games)
        awards.append(("💪", "Biggest Chad", w,
                       f"Beat {l} by {m:.1f} ({ws:.1f}–{ls:.1f})"))
        m, w, ws, l, ls = min(games)
        awards.append(("🍀", "Lucky Bastard", w,
                       f"Snuck past {l} by {m:.1f} ({ws:.1f}–{ls:.1f})"))

    # Starters' actual vs ESPN projection.
    wk = lw.filter(pl.col("week") == week).with_columns(pl.col("team_name").str.strip_chars())
    starters = wk.filter(~pl.col("slot").is_in(["BE", "IR"]))
    vs_proj = (starters.group_by("team_name")
               .agg(actual=pl.col("points").sum(), proj=pl.col("projected_points").sum())
               .with_columns(diff=pl.col("actual") - pl.col("proj"))
               .sort("diff"))
    if vs_proj.height:
        top = vs_proj.row(-1, named=True)
        awards.append(("🚀", "Never Tell Me the Odds", top["team_name"],
                       f"Scored {top['actual']:.1f} vs {top['proj']:.1f} projected "
                       f"({top['diff']:+.1f})"))
        bot = vs_proj.row(0, named=True)
        awards.append(("💩", "You FUCKING suck", bot["team_name"],
                       f"Scored {bot['actual']:.1f} vs {bot['proj']:.1f} projected "
                       f"({bot['diff']:+.1f})"))

    # Best score left out of a lineup (bench or IR).
    bench = wk.filter(pl.col("slot").is_in(["BE", "IR"])).sort("points", descending=True)
    if bench.height:
        b = bench.row(0, named=True)
        awards.append(("🔥", "Red Hot Benchwarmer", b["name"],
                       f"{b['points']:.1f} pts on {b['team_name']}'s bench "
                       f"({b['position']})"))
    return week, awards


def render_awards():
    week, awards = weekly_awards()
    if not awards:
        st.subheader("Weekly awards")
        st.caption("Awards appear once the first week is in the books.")
        return
    st.subheader(f"Week {week} awards")
    # Five across on desktop; Streamlit stacks the columns on narrow screens.
    for col, (emoji, title, winner, detail) in zip(st.columns(len(awards)), awards):
        with col, st.container(border=True):
            st.markdown(f"<div style='font-size:2rem;line-height:1'>{emoji}</div>"
                        f"<div style='font-weight:800;margin:0.4rem 0 0.25rem'>"
                        f"{html.escape(title)}</div>"
                        f"<div style='font-size:1.05rem'>{html.escape(winner)}</div>",
                        unsafe_allow_html=True)
            st.caption(detail)


def win_prob_chart(odds, col):
    """One bar per game split at the home team's win probability. Home on the
    left (slot 1), away on the right (slot 2), 50% rule for reference.

    Team names sit ON the bar -- home above its left end, away below its right
    end -- instead of in a y-axis gutter, so the bar gets the full width at any
    screen size (a phone included) and two long names can never collide."""
    dark = getattr(st.context.theme, "type", None) == "dark"
    ink = "#f0efec" if dark else "#31333f"
    rows, games = [], []
    for i, r in enumerate(odds.iter_rows(named=True)):
        p = r[col]
        home, away = r["home_team"].strip(), r["away_team"].strip()
        games.append({"Game": i, "Home": home, "Away": away})
        rows += [
            {"Game": i, "Team": home, "Side": "Home", "start": 0.0, "end": p, "Win %": p},
            {"Game": i, "Team": away, "Side": "Away", "start": p, "end": 1.0, "Win %": 1 - p},
        ]
    pdf = pl.DataFrame(rows).to_pandas()
    gdf = pl.DataFrame(games).to_pandas()
    y = alt.Y("Game:O", axis=None, scale=alt.Scale(paddingInner=0, paddingOuter=0))
    x_scale = alt.Scale(domain=[0, 1], nice=False)

    bars = alt.Chart(pdf).mark_bar(size=20, cornerRadius=4, stroke=chart_surface(),
                                   strokeWidth=2).encode(
        y=y,
        x=alt.X("start:Q", title="Win probability", scale=x_scale,
                axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1], grid=False)),
        x2="end:Q",
        color=alt.Color("Side:N", scale=alt.Scale(domain=["Home", "Away"], range=PALETTE[:2]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Team", alt.Tooltip("Win %:Q", format=".0%")],
    )
    mid = alt.Chart(pl.DataFrame({"x": [0.5]}).to_pandas()).mark_rule(
        color=ink, strokeDash=[3, 3], opacity=0.45).encode(x=alt.X("x:Q", scale=x_scale))
    pcts = alt.Chart(pdf).transform_filter(alt.datum["Win %"] >= 0.12).transform_calculate(
        mid="(datum.start + datum.end) / 2").mark_text(
        color="white", fontWeight="bold", fontSize=12).encode(
        y=y, x=alt.X("mid:Q", scale=x_scale), text=alt.Text("Win %:Q", format=".0%"))
    home_names = alt.Chart(gdf).transform_calculate(x="0").mark_text(
        align="left", baseline="bottom", dy=-13, color=ink, fontSize=13).encode(
        y=y, x=alt.X("x:Q", scale=x_scale), text="Home:N")
    away_names = alt.Chart(gdf).transform_calculate(x="1").mark_text(
        align="right", baseline="top", dy=13, color=ink, fontSize=13).encode(
        y=y, x=alt.X("x:Q", scale=x_scale), text="Away:N")
    return (bars + mid + pcts + home_names + away_names).properties(height=78 * len(games))


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
    chart = alt.Chart(pdf).mark_bar(cornerRadiusEnd=4, stroke=chart_surface(),
                                    strokeWidth=2).encode(
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


apply_theme()

if st.query_params.get("view") == "moves":
    render_moves_feed()
    st.stop()

st.title("🏈 Gun Squad League Dashboard")
render_ticker()
league_tab, projections_tab, accuracy_tab = st.tabs(["League", "Projections", "Model accuracy"])
with league_tab:
    render_league()
with projections_tab:
    render_projections()
with accuracy_tab:
    render_accuracy()
