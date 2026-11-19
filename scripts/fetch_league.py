"""
fetch_league.py

Pulls your ESPN league data and saves it for the dashboard:
    data/standings.parquet      team records + points for/against
    data/rosters.parquet        every team's roster
    data/matchups.parquet       current-week head-to-head (in-season only)
    data/matchups_last.parquet  the last completed week's results (for the awards)
    data/schedule.parquet       every team's weekly opponent + result
    data/league_weekly.parquet  player-level box scores, every week so far
    data/transactions.parquet   every waiver / trade / add / drop this season

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

import json
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


# ESPN activity message types -> what happened. 179/181/239 are all drops
# (plain, waiver-related, and commissioner/lineup-lock flavors).
MOVE_TYPES = {178: "FA ADDED", 180: "WAIVER ADDED", 179: "DROPPED",
              181: "DROPPED", 239: "DROPPED", 244: "TRADED"}


def _activity_page(league, offset, size=50):
    """One page of the league's transaction log, newest first. Same request
    espn-api's recent_activity() makes, minus its per-player lookups (one extra
    ESPN call for every dropped player -- minutes for a full season)."""
    filters = {"topics": {
        "filterType": {"value": ["ACTIVITY_TRANSACTIONS"]},
        "limit": size, "limitPerMessageSet": {"value": 25}, "offset": offset,
        "sortMessageDate": {"sortPriority": 1, "sortAsc": False},
        "sortFor": {"sortPriority": 2, "sortAsc": False},
        "filterIncludeMessageTypeIds": {"value": list(MOVE_TYPES)},
    }}
    data = league.espn_request.league_get(
        extend="/communication/", params={"view": "kona_league_communication"},
        headers={"x-fantasy-filter": json.dumps(filters)})
    return data.get("topics", [])


def build_transactions(league, max_pages=40):
    """The league's FULL transaction log: every add, drop, waiver claim and
    trade this season. One row per player moved; rows sharing a `move_id` were
    one transaction (e.g. an add + the drop that made room). Empty on a
    brand-new league, so we skip cleanly if there's nothing yet."""
    teams = {t.team_id: t.team_name for t in league.teams}
    rows = []
    try:
        for page in range(max_pages):
            topics = _activity_page(league, offset=page * 50)
            for topic in topics:
                for msg in topic.get("messages", []):
                    kind = MOVE_TYPES.get(msg.get("messageTypeId"))
                    if kind is None:
                        continue
                    if kind == "TRADED":
                        team_id, other_id = msg.get("from"), msg.get("to")
                    elif msg["messageTypeId"] == 239:
                        team_id, other_id = msg.get("for"), None
                    else:
                        team_id, other_id = msg.get("to"), None
                    rows.append({
                        "move_id": topic["id"],
                        "date": topic["date"],   # epoch ms; formatted at display time
                        "team_id": team_id,
                        "team": teams.get(team_id, str(team_id)),
                        "action": kind,
                        "espn_id": msg.get("targetId"),
                        "player": league.player_map.get(msg.get("targetId"),
                                                        str(msg.get("targetId"))),
                        # trades: the team receiving the player
                        "to_team": teams.get(other_id) if other_id is not None else None,
                        # waivers: the FAAB bid ESPN stores in `from`
                        "bid": msg.get("from") if kind == "WAIVER ADDED" else None,
                    })
            if len(topics) < 50:
                break
    except Exception as e:
        print(f"  transactions unavailable ({e.__class__.__name__}: {e})")
        return None
    if not rows:
        return None
    return pl.DataFrame(rows, infer_schema_length=None).sort("date", descending=True)


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

    # The awards need the last FINISHED week, which isn't the same as the
    # "current" week -- ESPN rolls current_week forward as soon as a week ends.
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