"""
fetch_projections.py

Pulls Sleeper's projections and saves them with real player names attached:
    data/projections_season.parquet   (full-year totals -- draft/preseason view)
    data/projections_weekly.parquet   (the current week -- in-season view)

Two things make this simpler than the old per-player approach:
  1. Sleeper has *bulk* projection endpoints that return every player in one
     call, so there's no looping and no rate-limiting to worry about.
  2. Sleeper only gives us a player_id and some numbers. We attach real names
     by joining to the crosswalk we built in fetch_crosswalk.py -- which is the
     whole reason that script runs first.

Sleeper's projection API needs no login, so this is safe to run from anywhere.
"""

from pathlib import Path

import polars as pl
import requests

DATA = Path(__file__).parent.parent / "data"
DATA.mkdir(exist_ok=True)

# Sleeper's "state" endpoint tells us the current season and week, so this
# script keeps working week to week without us editing anything.
STATE_URL = "https://api.sleeper.app/v1/state/nfl"

# The bulk projection endpoints live on api.sleeper.com (a different host than
# the state endpoint -- that's expected, not a typo).
PROJ_BASE = "https://api.sleeper.com/projections/nfl"

# Sleeper's full player list -- a big (~5MB) dump of every player. We use it
# only as a fallback to name the handful of very-recent players the crosswalk
# hasn't caught up on. Sleeper asks callers not to hit this more than once a
# day, which fits our daily schedule exactly.
PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]

# Sleeper reports projected points under a few scoring systems. We grab all
# three so you can pick whichever matches your league later without refetching:
#   pts_std       = standard (no points per reception)
#   pts_half_ppr  = half point per reception
#   pts_ppr       = full point per reception
SCORING = ["pts_std", "pts_half_ppr", "pts_ppr"]


def get_state():
    r = requests.get(STATE_URL, timeout=30)
    r.raise_for_status()
    return r.json()


def load_sleeper_players():
    """Load Sleeper's full player list as a fallback name source.

    Returns a Polars frame of player_id -> name/position/team. This is only
    used to fill gaps the crosswalk leaves; the crosswalk stays primary.
    """
    r = requests.get(PLAYERS_URL, timeout=60)
    r.raise_for_status()
    data = r.json()  # a dict keyed by player_id
    records = []
    for pid, p in data.items():
        if not isinstance(p, dict):
            continue
        name = p.get("full_name") or " ".join(
            part for part in [p.get("first_name"), p.get("last_name")] if part
        ) or None
        records.append({
            "player_id": str(pid),
            "s_name": name,
            "s_position": p.get("position"),
            "s_team": p.get("team"),
        })
    return pl.DataFrame(records)


def fetch_projections(url):
    """Hit a bulk projection endpoint and return the raw list of records."""
    # requests turns this list of tuples into repeated query params like
    # position[]=QB&position[]=RB&... which is what Sleeper expects.
    params = [("season_type", "regular"), ("order_by", "pts_ppr")]
    params += [("position[]", p) for p in POSITIONS]
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def to_frame(rows):
    """Turn Sleeper's raw JSON into a tidy Polars table of id + points."""
    records = []
    for row in rows:
        stats = row.get("stats") or {}
        # player_id comes back as a string; we keep it that way so it matches
        # the crosswalk after we cast (see enrich()).
        rec = {"player_id": str(row.get("player_id"))}
        for s in SCORING:
            rec[s] = stats.get(s)
        if row.get("week") is not None:  # present for weekly, absent for season
            rec["week"] = row.get("week")
        records.append(rec)
    return pl.DataFrame(records)


def enrich(proj, sleeper_players):
    """Attach real name / position / team, label defenses, and fill gaps.

    Offensive players resolve via the crosswalk join. Team defenses use a
    non-numeric Sleeper id (a team code like "CHI") that the crosswalk doesn't
    cover, so the join leaves them blank. We detect those by their id shape and
    label them directly -- no team-name lookup needed, since "CHI D/ST" is
    unambiguous and correct regardless of how Sleeper abbreviates teams.

    Anyone still unnamed after that (very recent players missing from the
    crosswalk) gets filled from Sleeper's own player list via pl.coalesce,
    which picks the first non-null value -- so a name we already have always
    wins, and Sleeper only fills the blanks.
    """
    crosswalk = (
        pl.read_parquet(DATA / "playerids.parquet")
        # crosswalk sleeper_id is an integer (i64); Sleeper's player_id is a
        # string. Cast to string so the two sides can actually match.
        .with_columns(pl.col("sleeper_id").cast(pl.Utf8))
        .select(["sleeper_id", "name", "position", "team"])
    )
    # left join: keep every projected player, fill name/team where we have it.
    proj = proj.join(crosswalk, left_on="player_id", right_on="sleeper_id", how="left")

    # A defense's id is a team code (letters), not a number. This flag is true
    # for defenses and false for everyone else, whatever the exact code is.
    is_def = ~pl.col("player_id").str.contains(r"^\d+$")
    proj = proj.with_columns(
        pl.when(is_def).then(pl.lit("DEF")).otherwise(pl.col("position")).alias("position"),
        pl.when(is_def).then(pl.col("player_id")).otherwise(pl.col("team")).alias("team"),
        # Only overwrite name where the crosswalk left it blank, so we never
        # clobber a real player's name.
        pl.when(is_def & pl.col("name").is_null())
        .then(pl.col("player_id") + pl.lit(" D/ST"))
        .otherwise(pl.col("name"))
        .alias("name"),
    )

    # Fill any remaining blanks from Sleeper's player list. coalesce keeps the
    # crosswalk value when present and only reaches for Sleeper's when it's null.
    proj = proj.join(sleeper_players, on="player_id", how="left").with_columns(
        pl.coalesce("name", "s_name").alias("name"),
        pl.coalesce("position", "s_position").alias("position"),
        pl.coalesce("team", "s_team").alias("team"),
    ).drop("s_name", "s_position", "s_team")

    return proj


def keep_projected(proj):
    """Drop rows with no projection under any scoring system -- Sleeper lists a
    huge pool of players it doesn't actually project, and those empties don't
    belong in a projections file."""
    return proj.filter(pl.any_horizontal(pl.col(SCORING).is_not_null()))


def main():
    state = get_state()
    season = state["season"]
    week = state["week"]
    season_type = state.get("season_type")
    print(f"Sleeper state -> season={season}, week={week}, type={season_type}")

    # Load the fallback name source once, reused for season + weekly.
    sleeper_players = load_sleeper_players()

    # ---- Season-long projections (works year-round, incl. draft season) ----
    season_rows = fetch_projections(f"{PROJ_BASE}/{season}")
    if not season_rows:
        print("Season projections came back empty -- paste this output and "
              "we'll check the endpoint.")
    else:
        season_proj = keep_projected(enrich(to_frame(season_rows), sleeper_players))
        season_proj.write_parquet(DATA / "projections_season.parquet")
        print(f"Season: {season_proj.height} projected players -> "
              "data/projections_season.parquet")
        print(season_proj.head())

        # Sanity checks: defenses should be ~32, and nobody projected should
        # be left without a name now that the fallback is in place.
        defenses = season_proj.filter(pl.col("position") == "DEF")
        still_unnamed = season_proj.filter(pl.col("name").is_null()).height
        print(f"Defenses labeled: {defenses.height} | still unnamed: {still_unnamed}")

    # ---- Weekly projections (only meaningful during the regular season) ----
    if season_type == "regular" and week and week > 0:
        week_rows = fetch_projections(f"{PROJ_BASE}/{season}/{week}")
        week_proj = keep_projected(enrich(to_frame(week_rows), sleeper_players))
        week_proj.write_parquet(DATA / "projections_weekly.parquet")
        print(f"Week {week}: {week_proj.height} projected players -> "
              "data/projections_weekly.parquet")
        print(week_proj.head())
    else:
        print("Not in the regular season yet -- skipping weekly projections "
              "(they'll start populating once the season begins).")


if __name__ == "__main__":
    main()