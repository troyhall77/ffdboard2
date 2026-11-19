"""
fetch_league.py

Pulls your ESPN league data and saves it for the dashboard:
    data/standings.parquet      team records + points for/against
    data/rosters.parquet        every team's roster
    data/matchups.parquet       current-week head-to-head (in-season only)
    data/matchups_last.parquet  the last completed week's results (for the recap)
    data/schedule.parquet       every team's weekly opponent + result
    data/league_weekly.parquet  player-level box scores, every week so far
    data/transactions.parquet   recent waivers / trades / adds / drops

This is the ESPN half -- the part you already had working in R with ffscrapr.
Two things are new compared with the Sleeper scripts:
  1. It's authenticated. ESPN needs to know it's you to read a private league,
     so it uses two browser cookies (espn_s2 and SWID).
  2. It uses the `espn-api` package instead of raw requests -- that package
     wraps ESPN's messy endpoints in tidy Python objects.

Credentials are read from the environment. Locally that comes from a .env file
(never committed); in GitHub Actions it comes from encrypted secrets. Same code
either way.
"""

import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from espn_api.football import League

# Loads variables from a local .env file if one exists. In GitHub Actions there
# is no .env -- the variables are already set -- and load_dotenv() just does
# nothing, so this one line works in both places.
load_dotenv()

DATA = Path(__file__).parent.parent / "data"
DATA.mkdir(exist_ok=True)

LEAGUE_ID = int(os.environ["LEAGUE_ID"])
SEASON = int(os.environ["SEASON"])
ESPN_S2 = os.environ["ESPN_S2"]
SWID = os.environ["SWID"]


def owner_name(team):
    """Get a readable owner name across espn-api versions.

    Newer versions expose team.owners as a list of dicts; older ones expose
    team.owner as a plain string. We handle both so a version bump won't break
    the script."""
    owners = getattr(team, "owners", None)
    if owners and isinstance(owners, list) and isinstance(owners[0], dict):
        o = owners[0]
        return f"{o.get('firstName', '')} {o.get('lastName', '')}".strip() or None
    return getattr(team, "owner", None)


def build_standings(league):
    rows = [{
        "team_id": t.team_id,
        "team_name": t.team_name,
        "owner": owner_name(t),
        "wins": t.wins,
        "losses": t.losses,
        "ties": t.ties,
        "points_for": t.points_for,
        "points_against": t.points_against,
        "standing": t.standing,
    } for t in league.teams]
    return pl.DataFrame(rows)


def build_rosters(league):
    rows = []
    for t in league.teams:
        for p in t.roster:
            rows.append({
                "team_id": t.team_id,
                "team_name": t.team_name,
                # playerId is ESPN's integer id -- the same id our crosswalk
                # stores as espn_id, so rosters join cleanly to projections.
                "espn_id": p.playerId,
                "player_name": p.name,
                "position": p.position,
                "pro_team": p.proTeam,
                "lineup_slot": getattr(p, "lineupSlot", None),
            })
    return pl.DataFrame(rows) if rows else None


def build_matchups(league, week=None):
    """One week's head-to-head scores (current week by default). Doesn't exist
    in the offseason, so we skip cleanly rather than error out."""
    try:
        box = league.box_scores(week) if week else league.box_scores()
    except Exception as e:
        print(f"  matchups unavailable ({e.__class__.__name__}) -- likely offseason")
        return None
    rows = []
    for m in box:
        rows.append({
            "week": week or league.current_week,
            # home/away can be 0 on a bye, so guard with getattr.
            "home_team_id": getattr(m.home_team, "team_id", None),
            "home_team": getattr(m.home_team, "team_name", None),
            "home_score": m.home_score,
            "home_projected": getattr(m, "home_projected", None),
            "away_team_id": getattr(m.away_team, "team_id", None),
            "away_team": getattr(m.away_team, "team_name", None),
            "away_score": m.away_score,
            "away_projected": getattr(m, "away_projected", None),
        })
    return pl.DataFrame(rows) if rows else None


def build_schedule(league):
    """Every team's regular-season opponent, score and result by week. 'U' =
    undecided, which is how we tell finished weeks from upcoming ones."""
    rows = []
    for t in league.teams:
        for i, opp in enumerate(t.schedule):
            rows.append({
                "week": i + 1,
                "team_id": t.team_id,
                "opponent_id": getattr(opp, "team_id", None),
                "score": t.scores[i] if i < len(t.scores) else None,
                "outcome": t.outcomes[i] if i < len(t.outcomes) else "U",
            })
    return pl.DataFrame(rows) if rows else None


def last_completed_week(schedule):
    done = schedule.filter(pl.col("outcome") != "U")
    return done["week"].max() if done.height else None


def build_league_weekly(league):
    """Player-level box scores for every week so far, current week included.

    This is ESPN's view: actual points in OUR league's scoring, ESPN's own
    pregame projection, and the lineup slot each manager used. The backtest uses
    it to score ESPN's projections and to measure K / D/ST, which nflverse doesn't
    cover; the matchup sim uses the current week's lineups (and any games already
    final) to compute live win odds."""
    rows = []
    for wk in range(1, league.current_week + 1):
        try:
            box = league.box_scores(wk)
        except Exception as e:
            print(f"  box scores for week {wk} unavailable ({e.__class__.__name__})")
            continue
        for m in box:
            for side in ("home", "away"):
                team = getattr(m, f"{side}_team", None)
                if not hasattr(team, "team_id"):
                    continue   # bye
                for p in getattr(m, f"{side}_lineup", []):
                    rows.append({
                        "week": wk,
                        "team_id": team.team_id,
                        "team_name": team.team_name,
                        "slot": p.slot_position,
                        "espn_id": p.playerId,
                        "name": p.name,
                        "position": p.position,
                        "pro_team": p.proTeam,
                        "points": p.points,
                        "projected_points": p.projected_points,
                        # 0 = not started, 100 = final; in between = in progress.
                        "game_played": getattr(p, "game_played", None),
                    })
    return pl.DataFrame(rows) if rows else None


def build_transactions(league):
    """Recent waivers / trades / adds / drops. Also empty on a brand-new
    league, so we skip cleanly if there's nothing yet."""
    try:
        activity = league.recent_activity(size=25)
    except Exception as e:
        print(f"  transactions unavailable ({e.__class__.__name__})")
        return None
    rows = []
    for a in activity:
        for action in a.actions:
            team, act_type, player = action[0], action[1], action[2]
            rows.append({
                "date": a.date,  # epoch milliseconds; we'll format at display time
                "team": getattr(team, "team_name", str(team)),
                "action": act_type,
                "player": getattr(player, "name", str(player)),
            })
    return pl.DataFrame(rows) if rows else None


def main():
    print(f"Connecting to league {LEAGUE_ID}, season {SEASON}...")
    try:
        league = League(
            league_id=LEAGUE_ID, year=SEASON, espn_s2=ESPN_S2, swid=SWID
        )
    except Exception as e:
        print(f"Could not connect ({e.__class__.__name__}). Most likely the "
              "cookies are stale or the league id / season is off. Re-copy "
              "espn_s2 and SWID from your browser and check the .env values.")
        raise

    # Standings always exist once a league is created, so this is our proof
    # that the cookies work.
    standings = build_standings(league)
    standings.write_parquet(DATA / "standings.parquet")
    print(f"Standings: {standings.height} teams -> data/standings.parquet")
    print(standings.head())

    # The recap needs the last FINISHED week, which isn't the same as the
    # "current" week -- ESPN can roll current_week forward before Tuesday's job.
    schedule = build_schedule(league)
    last_week = last_completed_week(schedule) if schedule is not None else None
    last_matchups = build_matchups(league, last_week) if last_week else None

    s = league.settings
    settings = pl.DataFrame([{
        "current_week": league.current_week,
        "reg_season_count": s.reg_season_count,
        "playoff_team_count": s.playoff_team_count,
    }])

    # The rest may be empty depending on how far along the 2026 league is.
    for name, frame in [
        ("league_settings", settings),
        ("rosters", build_rosters(league)),
        ("matchups", build_matchups(league)),
        ("matchups_last", last_matchups),
        ("schedule", schedule),
        ("league_weekly", build_league_weekly(league)),
        ("transactions", build_transactions(league)),
    ]:
        if frame is None:
            print(f"{name}: nothing to write yet")
        else:
            frame.write_parquet(DATA / f"{name}.parquet")
            print(f"{name}: {frame.height} rows -> data/{name}.parquet")


if __name__ == "__main__":
    main()