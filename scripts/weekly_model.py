"""
weekly_model.py

The in-season weekly model, shared by backtest.py (which fits it) and
project_inseason.py (which uses it). Not run on its own.

The model in one paragraph: a player's projection for a week starts from
Sleeper's number and leans toward what he's actually averaged this season, by an
amount that grows with games played -- a "credibility" blend,

    proj = sleeper + g / (g + k) * (ppg - sleeper)

so with 0 games it's pure Sleeper, and a player with g = k games sits halfway.
k is fit per position by the backtest (small k = trust actual production fast).
Around that mean, weekly points follow the EMPIRICAL distribution of
actual / projected seen this season for that position and projection tier (the
backtest stores its quantiles). That captures the real shape -- lots of
duds, a long boom tail -- and any systematic over/under-projection, which a
textbook gamma curve doesn't. On top of that is a chance he doesn't play.

Everything is in PPR, which is this league's scoring.
"""

from pathlib import Path

import numpy as np
import polars as pl

DATA = Path(__file__).parent.parent / "data"

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]
OFFENSE = ["QB", "RB", "WR", "TE"]

# Projection-tier floors (weekly PPR points) for the CV lookup. A player uses the
# highest floor his projection clears. Fixed cutoffs rather than quantiles so the
# tiers read as "a 15-point RB", not "RB tier 3".
TIER_FLOORS = {
    "QB": [0, 12, 16, 20],
    "RB": [0, 5, 9, 13, 17],
    "WR": [0, 5, 9, 13, 17],
    "TE": [0, 4, 7, 10],
    "K": [0],
    "DEF": [0],
}

# Quantile grid for the stored actual/projected ratio distribution.
RATIO_GRID = [1, *range(5, 100, 5), 99]

# Must match the app's PCT_COLS.
PCTS = list(range(5, 100, 5))
PCT_COLS = [f"p{p:02d}" for p in PCTS]

# ESPN and Sleeper disagree on exactly one team code.
ESPN_TO_SLEEPER_TEAM = {"WSH": "WAS"}


# ---------------------------------------------------------------- id mapping

def espn_to_sleeper(frame):
    """Add a Sleeper `player_id` to an ESPN frame (espn_id, position, pro_team).
    Players map through the crosswalk; a D/ST's Sleeper id IS its team code."""
    crosswalk = (
        pl.read_parquet(DATA / "playerids.parquet")
        .select("espn_id", pl.col("sleeper_id").cast(pl.Utf8).alias("xw_id"))
        .drop_nulls()
        .unique("espn_id")
    )
    team_code = pl.col("pro_team").replace(ESPN_TO_SLEEPER_TEAM)
    return (
        frame.join(crosswalk, on="espn_id", how="left")
        .with_columns(
            pl.when(pl.col("position") == "D/ST").then(team_code)
            .otherwise(pl.col("xw_id")).alias("player_id"),
            pl.when(pl.col("position") == "D/ST").then(pl.lit("DEF"))
            .otherwise(pl.col("position")).alias("position"),
        )
        .drop("xw_id")
    )


# ---------------------------------------------------------------- features

def prior_production(actuals, season):
    """Per player-week: games played and points-per-game in the season BEFORE
    that week -- so week 7's number only knows about weeks 1-6 (no peeking).

    Returns (player_id, week, g_prior, ppg_prior) for weeks 1..19, where week
    w+1's row reflects everything through week w."""
    act = actuals.filter(pl.col("season") == season).select("player_id", "week", "pts_ppr")
    weeks = pl.DataFrame({"week": list(range(1, 20))}, schema={"week": pl.Int64})
    grid = act.select("player_id").unique().join(weeks, how="cross")
    return (
        grid.join(act.rename({"week": "w_act"}), on="player_id", how="left")
        .with_columns(before=pl.col("w_act") < pl.col("week"))
        .group_by("player_id", "week")
        .agg(
            g_prior=pl.col("before").sum().cast(pl.Int64),
            pts_prior=pl.when(pl.col("before")).then(pl.col("pts_ppr")).otherwise(0).sum(),
        )
        .with_columns(
            ppg_prior=pl.when(pl.col("g_prior") > 0)
            .then(pl.col("pts_prior") / pl.col("g_prior"))
        )
        .drop("pts_prior")
    )


def blend_expr(k):
    """Credibility blend as a Polars expression. `k` may be a column or number.
    Falls back to pure Sleeper when there's no production yet."""
    g = pl.col("g_prior").fill_null(0)
    lam = g / (g + k)
    return (
        pl.when(pl.col("ppg_prior").is_null() | (g == 0)).then(pl.col("sleeper"))
        .otherwise(pl.col("sleeper") + lam * (pl.col("ppg_prior") - pl.col("sleeper")))
    )


# ---------------------------------------------------------------- params

def load_params():
    """weekly_params.parquet ->
    {position: {"k", "dnp", "talent_sd", "tiers": [(floor, cv, ratio_q), ...]}}"""
    params = pl.read_parquet(DATA / "weekly_params.parquet")
    out = {}
    for pos in params["position"].unique().to_list():
        sub = params.filter(pl.col("position") == pos).sort("proj_floor", descending=True)
        first = sub.row(0, named=True)
        out[pos] = {
            "k": first["k"],
            "dnp": first["dnp_rate"],
            "talent_sd": first["talent_sd"],
            "tiers": list(zip(sub["proj_floor"].to_list(), sub["cv"].to_list(),
                              [np.array(q) for q in sub["ratio_q"].to_list()])),
        }
    return out


def tier_for(params, position, proj):
    """(cv, ratio quantiles) of the tier a projection falls in."""
    tiers = params[position]["tiers"]
    for floor, cv, rq in tiers:
        if proj >= floor:
            return cv, rq
    return tiers[-1][1], tiers[-1][2]


# ---------------------------------------------------------------- sampling

def sample_ratio(ratio_q, rng, size):
    """Inverse-CDF draws from a stored ratio distribution (piecewise linear
    between the saved quantiles; tails clipped at p01 / p99)."""
    return np.interp(rng.random(size), np.array(RATIO_GRID) / 100, ratio_q)


def sample_week(mean, ratio_q, p_play, rng, n):
    """n draws of one player-week: projection x a realistic actual/projected
    ratio if he plays, else 0."""
    if mean <= 0:
        return np.zeros(n)
    return mean * sample_ratio(ratio_q, rng, n) * (rng.random(n) < p_play)


def percentile_record(samples):
    qs = np.percentile(samples, PCTS)
    rec = {c: round(float(v), 1) for c, v in zip(PCT_COLS, qs)}
    rec["sim_floor"], rec["sim_median"], rec["sim_ceiling"] = rec["p10"], rec["p50"], rec["p90"]
    rec["sim_mean"] = round(float(samples.mean()), 1)
    return rec
