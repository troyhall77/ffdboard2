"""
project_inseason.py

Runs the fitted weekly model (weekly_model.py + data/weekly_params.parquet from
backtest.py) forward to produce everything in-season:

    data/projections_week_sim.parquet   this week: floor/median/ceiling per player
    data/projections_ros.parquet        rest of season: points so far + simulated
                                        remaining weeks = final-total range
    data/matchup_odds.parquet           this week's league matchups: win odds
    data/playoff_odds.parquet           every remaining week simmed: playoff odds

How the rest-of-season sim differs from the weekly one:
  * Each remaining week uses Sleeper's projection for THAT week, so byes (no
    projection) count as zeros automatically.
  * Injuries persist. A simulated player who gets hurt stays out for a stretch
    (average 3 weeks) instead of flipping a fresh coin every week, so his range
    widens the way a real injury would. The rate is tuned to match 2025's
    games-played (data/variance_params.parquet).
  * A player-level "projection was off" factor, shared across his weeks, uses
    the spread the backtest measured. (It came out small: Sleeper's errors
    don't persist much week to week.)

Matchup odds use each manager's actual starting lineup and are live: a player
whose game is final counts his real points; one still to play is simulated.

Reads:  data/weekly_params.parquet, data/projections_weekly_all.parquet,
        data/actuals_weekly.parquet, data/variance_params.parquet,
        data/league_weekly.parquet, data/rosters.parquet, data/schedule.parquet,
        data/standings.parquet, data/league_settings.parquet
"""

import os

import numpy as np
import polars as pl
from dotenv import load_dotenv

from weekly_model import (DATA, OFFENSE, blend_expr, espn_to_sleeper, load_params,
                          percentile_record, prior_production, sample_ratio,
                          sample_week, tier_for)

load_dotenv()
SEASON = int(os.environ.get("SEASON", 2026))
SEED = 11
N_PLAYER = 5000         # sims per player (weekly + ROS)
N_LEAGUE = 10000        # sims per team-week (matchups + playoffs)
LAST_NFL_WEEK = 18
INJURY_WEEKS = 3.0      # average length of a simulated injury absence

LINEUP = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1}
FLEX_OK = {"RB", "WR", "TE"}
BENCH_SLOTS = {"BE", "IR"}


# ---------------------------------------------------------------- inputs

def model_frame(params, current_week):
    """Model mean (`proj`) for every player-week from the current week on, using
    production through the last finished week."""
    sleeper = (
        pl.read_parquet(DATA / "projections_weekly_all.parquet")
        .filter((pl.col("season") == SEASON) & (pl.col("week") >= current_week)
                & (pl.col("pts_ppr") > 0) & pl.col("position").is_in(list(params)))
        .select("player_id", "name", "position", "team", "week",
                pl.col("pts_ppr").alias("sleeper"))
    )
    actuals = pl.read_parquet(DATA / "actuals_weekly.parquet")
    prior = (prior_production(actuals, SEASON)
             .filter(pl.col("week") == current_week).drop("week"))
    k = pl.DataFrame({"position": list(params), "k": [p["k"] for p in params.values()]})
    return (
        sleeper.join(prior, on="player_id", how="left")
        .join(k, on="position")
        .with_columns(proj=blend_expr(pl.col("k")))
    )


def season_availability():
    """Per-week chance a player is healthy, from 2025 games played."""
    va = pl.read_parquet(DATA / "variance_params.parquet")
    out = dict(va.group_by("position").agg(pl.col("availability").first()).iter_rows())
    out.update({"K": 1.0, "DEF": 1.0})
    return out


def player_meta(frame):
    """Name / position / team for anyone we might output, preferring Sleeper's
    current listing and falling back to the nflverse box score."""
    actuals = pl.read_parquet(DATA / "actuals_weekly.parquet").sort("season", "week", descending=True)
    return pl.concat([
        frame.select("player_id", "name", "position", "team"),
        actuals.select("player_id", "name", "position", "team"),
    ]).unique("player_id", keep="first")


# ---------------------------------------------------------------- B: this week

def weekly_sims(frame, params, week, rng):
    rows = []
    for r in frame.filter(pl.col("week") == week).iter_rows(named=True):
        p = params[r["position"]]
        _, rq = tier_for(params, r["position"], r["proj"])
        draws = sample_week(r["proj"], rq, 1 - p["dnp"], rng, N_PLAYER)
        rows.append({
            "player_id": r["player_id"], "week": week,
            "sleeper_proj": r["sleeper"], "model_proj": round(r["proj"], 2),
            "games": r["g_prior"], "ppg": r["ppg_prior"],
            **percentile_record(draws),
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- A: rest of season

def ros_sims(frame, params, avail, current_week, rng):
    actuals = pl.read_parquet(DATA / "actuals_weekly.parquet").filter(
        (pl.col("season") == SEASON) & (pl.col("week") < current_week))
    ytd = actuals.group_by("player_id").agg(
        ytd_pts=pl.col("pts_ppr").sum(), ytd_games=pl.len())

    future = frame.filter(pl.col("position").is_in(OFFENSE))
    by_player = future.group_by("player_id").agg(
        pl.col("position").first(), pl.col("week"), pl.col("proj"))
    players = by_player.join(ytd, on="player_id", how="full", coalesce=True)

    weeks = list(range(current_week, LAST_NFL_WEEK + 1))
    rows = []
    for r in players.iter_rows(named=True):
        ytd_pts = r["ytd_pts"] or 0.0
        total = np.full(N_PLAYER, ytd_pts)
        ros_weeks = 0
        if r["week"]:
            pos = r["position"]
            p = params[pos]
            proj_by_week = dict(zip(r["week"], r["proj"]))
            # Injury as a 2-state chain: healthy -> out with prob a, out ->
            # healthy with prob b. Stationary share out = a / (a + b) = 1 - avail.
            b = 1.0 / INJURY_WEEKS
            a = (1 - avail[pos]) * b / avail[pos]
            out = rng.random(N_PLAYER) < p["dnp"]          # this week: known status
            sd = p["talent_sd"]
            skill = np.exp(rng.normal(-sd * sd / 2, sd, N_PLAYER)) if sd > 0 else 1.0
            for w in weeks:
                if w > current_week:
                    u = rng.random(N_PLAYER)
                    out = np.where(out, u >= b, u < a)
                mean = proj_by_week.get(w, 0.0)
                if mean <= 0:
                    continue                                  # bye / not projected
                ros_weeks += 1
                _, rq = tier_for(params, pos, mean)
                total += skill * mean * sample_ratio(rq, rng, N_PLAYER) * ~out
        rec = {"player_id": r["player_id"], "ytd_pts": round(ytd_pts, 1),
               "ytd_games": r["ytd_games"] or 0, "ros_weeks": ros_weeks}
        rec.update(percentile_record(total))
        rec["ros_mean"] = round(rec["sim_mean"] - ytd_pts, 1)
        rows.append(rec)
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- C: matchups

def player_sampler(params):
    """Returns draw(position, mean, p_play, rng, n) using the tier's ratio distribution."""
    def draw(position, mean, p_play, rng, n):
        if position not in params or mean is None or mean <= 0:
            return np.zeros(n)
        _, rq = tier_for(params, position, mean)
        return sample_week(mean, rq, p_play, rng, n)
    return draw


def lineup_samples(frame, params, week, rng):
    """Team score samples for the current week from the actual starting lineups."""
    lw = pl.read_parquet(DATA / "league_weekly.parquet").filter(
        (pl.col("week") == week) & ~pl.col("slot").is_in(list(BENCH_SLOTS)))
    lw = espn_to_sleeper(lw).join(
        frame.filter(pl.col("week") == week).select("player_id", "proj"),
        on="player_id", how="left")
    draw = player_sampler(params)
    samples = {}
    for (team_id,), grp in lw.group_by(["team_id"]):
        total = np.zeros(N_LEAGUE)
        for r in grp.iter_rows(named=True):
            gp = r["game_played"] or 0
            pts = r["points"] or 0.0
            if gp >= 100:
                total += pts                       # final (or on bye: 0)
                continue
            mean = r["proj"] if r["proj"] is not None else r["projected_points"]
            p_play = 1 - params.get(r["position"], {}).get("dnp", 0.0)
            remaining = draw(r["position"], mean, p_play, rng, N_LEAGUE)
            total += pts + (1 - gp / 100) * remaining  # in progress: scale the rest
        samples[team_id] = total
    return samples


def best_lineup_samples(roster, week_frame, params, avail, rng):
    """Score samples if the team starts its highest-projected legal lineup.
    `avail` = chance each position plays (season availability for future weeks,
    1 - DNP rate for this week)."""
    pool = (roster.join(week_frame.select("player_id", "proj"), on="player_id", how="left")
            .with_columns(pl.col("proj").fill_null(0.0))
            .sort("proj", descending=True))
    used, starters = set(), []
    for slot, count in LINEUP.items():
        ok = FLEX_OK if slot == "FLEX" else {slot}
        for r in pool.iter_rows(named=True):
            if count == 0:
                break
            if r["player_id"] in used or r["position"] not in ok:
                continue
            used.add(r["player_id"])
            starters.append(r)
            count -= 1
    draw = player_sampler(params)
    total = np.zeros(N_LEAGUE)
    for r in starters:
        total += draw(r["position"], r["proj"], avail.get(r["position"], 1.0), rng, N_LEAGUE)
    return total


def load_rosters():
    return espn_to_sleeper(
        pl.read_parquet(DATA / "rosters.parquet").filter(pl.col("lineup_slot") != "IR")
    ).select("team_id", "player_id", "position").drop_nulls("player_id")


def win_prob(h, a):
    return float((h > a).mean() + 0.5 * (h == a).mean())


def matchup_odds(samples, best_samples, week):
    """Two views of each game: the lineups as currently set (live -- includes
    any bye-week starters a manager hasn't swapped out yet), and if both
    managers started their best available lineup."""
    m = pl.read_parquet(DATA / "matchups.parquet")
    rows = []
    for r in m.iter_rows(named=True):
        h, a = samples.get(r["home_team_id"]), samples.get(r["away_team_id"])
        if h is None or a is None:
            continue
        hb, ab = best_samples.get(r["home_team_id"]), best_samples.get(r["away_team_id"])
        rows.append({
            "week": week, "home_team": r["home_team"], "away_team": r["away_team"],
            "home_score": r["home_score"], "away_score": r["away_score"],
            "home_mean": float(h.mean()), "away_mean": float(a.mean()),
            "home_p10": float(np.percentile(h, 10)), "home_p90": float(np.percentile(h, 90)),
            "away_p10": float(np.percentile(a, 10)), "away_p90": float(np.percentile(a, 90)),
            "home_win_prob": win_prob(h, a),
            "home_best_mean": float(hb.mean()), "away_best_mean": float(ab.mean()),
            "home_win_prob_best": win_prob(hb, ab),
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------- playoffs

def playoff_odds(current_samples, frame, params, avail, current_week, rng):
    settings = pl.read_parquet(DATA / "league_settings.parquet").row(0, named=True)
    standings = pl.read_parquet(DATA / "standings.parquet")
    schedule = pl.read_parquet(DATA / "schedule.parquet")
    remaining = schedule.filter((pl.col("outcome") == "U") & (pl.col("week") >= current_week)
                                & (pl.col("team_id") < pl.col("opponent_id")))
    if not remaining.height:
        return None

    rosters = load_rosters()

    team_ids = standings["team_id"].to_list()
    idx = {t: i for i, t in enumerate(team_ids)}
    wins = np.tile((standings["wins"] + 0.5 * standings["ties"]).to_numpy(), (N_LEAGUE, 1)).astype(float)
    pf = np.tile(standings["points_for"].to_numpy(), (N_LEAGUE, 1)).astype(float)

    for wk in sorted(remaining["week"].unique().to_list()):
        if wk == current_week:
            # This week uses lineups as set: it's what will actually score, and
            # once games are final it IS the result.
            week_samples = current_samples
        else:
            wf = frame.filter(pl.col("week") == wk)
            week_samples = {
                t: best_lineup_samples(rosters.filter(pl.col("team_id") == t), wf,
                                       params, avail, rng)
                for t in team_ids
            }
        for g in remaining.filter(pl.col("week") == wk).iter_rows(named=True):
            h, a = week_samples[g["team_id"]], week_samples[g["opponent_id"]]
            i, j = idx[g["team_id"]], idx[g["opponent_id"]]
            wins[:, i] += (h > a) + 0.5 * (h == a)
            wins[:, j] += (a > h) + 0.5 * (h == a)
            pf[:, i] += h
            pf[:, j] += a

    # Rank on wins, then points for. (ESPN's real tiebreak may differ; PF is the
    # common default and ties on wins are where it matters.)
    key = wins * 1e5 + pf
    seed = (-key).argsort(axis=1).argsort(axis=1) + 1        # 1 = top seed
    n_playoff = settings["playoff_team_count"]
    return pl.DataFrame({
        "team_id": team_ids,
        "team_name": standings["team_name"].to_list(),
        "proj_wins": wins.mean(axis=0).round(1),
        "playoff_pct": (seed <= n_playoff).mean(axis=0),
        "top_seed_pct": (seed == 1).mean(axis=0),
        "last_place_pct": (seed == len(team_ids)).mean(axis=0),
        "avg_seed": seed.mean(axis=0).round(1),
    }).sort("playoff_pct", descending=True)


# ---------------------------------------------------------------- main

def main():
    needed = ["weekly_params", "projections_weekly", "league_settings", "matchups"]
    missing = [n for n in needed if not (DATA / f"{n}.parquet").exists()]
    if missing:
        print(f"Not in season yet (missing {', '.join(missing)}) -- skipping.")
        return
    params = load_params()
    avail = season_availability()
    rng = np.random.default_rng(SEED)

    current_week = int(pl.read_parquet(DATA / "projections_weekly.parquet")["week"].max())
    league_week = int(pl.read_parquet(DATA / "league_settings.parquet")["current_week"][0])
    frame = model_frame(params, current_week)
    print(f"Sleeper week {current_week} | ESPN week {league_week} | "
          f"{frame.height} player-weeks to project")

    meta = player_meta(frame)
    week_sim = meta.join(weekly_sims(frame, params, current_week, rng), on="player_id")
    week_sim.write_parquet(DATA / "projections_week_sim.parquet")
    print(f"Weekly: {week_sim.height} players -> data/projections_week_sim.parquet")

    ros = meta.join(ros_sims(frame, params, avail, current_week, rng), on="player_id")
    ros.write_parquet(DATA / "projections_ros.parquet")
    print(f"Rest of season: {ros.height} players -> data/projections_ros.parquet")

    # League sims key off ESPN's week (the one the lineups are set for).
    league_frame = frame if league_week == current_week else model_frame(params, league_week)
    samples = lineup_samples(league_frame, params, league_week, rng)
    rosters = load_rosters()
    this_week = league_frame.filter(pl.col("week") == league_week)
    healthy_now = {pos: 1 - p["dnp"] for pos, p in params.items()}
    best = {t: best_lineup_samples(rosters.filter(pl.col("team_id") == t), this_week,
                                   params, healthy_now, rng)
            for t in rosters["team_id"].unique().to_list()}
    odds = matchup_odds(samples, best, league_week)
    odds.write_parquet(DATA / "matchup_odds.parquet")
    print(f"Matchups: {odds.height} games -> data/matchup_odds.parquet")
    print(odds.select("home_team", "home_mean", "away_mean", "away_team", "home_win_prob"))

    po = playoff_odds(samples, league_frame, params, avail, league_week, rng)
    if po is not None:
        po.write_parquet(DATA / "playoff_odds.parquet")
        print("Playoff odds -> data/playoff_odds.parquet")
        print(po)
    else:
        print("Regular season is over -- no playoff odds to compute.")


if __name__ == "__main__":
    main()
