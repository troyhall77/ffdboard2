"""
fetch_actuals.py

Downloads real weekly fantasy results from nflverse and saves them keyed by
Sleeper player id, so they line up with the projections:
    data/actuals_weekly.parquet

Covers last season (full) plus the current season so far, regular season only.
nflverse already computes standard and PPR points; half-PPR is just their
midpoint, so we derive it rather than scoring anything ourselves.

One row per player per game PLAYED. A player who sat out has no row -- the
backtest relies on that to measure how often projected players don't play.

K and D/ST aren't scored by nflverse, so they come from ESPN box scores instead
(see league_weekly.parquet from fetch_league.py).

Reads:  data/playerids.parquet (needs gsis_id -- run fetch_crosswalk.py first)
Writes: data/actuals_weekly.parquet
"""

import os
from pathlib import Path

import nflreadpy as nfl
import polars as pl
from dotenv import load_dotenv

load_dotenv()

DATA = Path(__file__).parent.parent / "data"
SEASON = int(os.environ.get("SEASON", 2026))
POSITIONS = ["QB", "RB", "WR", "TE"]


def main():
    stats = (
        nfl.load_player_stats(seasons=[SEASON - 1, SEASON])
        .filter((pl.col("season_type") == "REG") & pl.col("position").is_in(POSITIONS))
    )

    crosswalk = (
        pl.read_parquet(DATA / "playerids.parquet")
        .select(pl.col("gsis_id"), pl.col("sleeper_id").cast(pl.Utf8))
        .drop_nulls()
        .unique("gsis_id")
    )

    # nflverse's own `player_id` column IS the gsis id; we swap it for Sleeper's.
    actuals = (
        stats.join(crosswalk, left_on="player_id", right_on="gsis_id", how="inner")
        .select(
            pl.col("sleeper_id").alias("player_id"),
            pl.col("player_display_name").alias("name"),
            "position",
            "team",
            pl.col("season").cast(pl.Int64),
            pl.col("week").cast(pl.Int64),
            pl.col("fantasy_points").alias("pts_std"),
            ((pl.col("fantasy_points") + pl.col("fantasy_points_ppr")) / 2).alias("pts_half_ppr"),
            pl.col("fantasy_points_ppr").alias("pts_ppr"),
        )
        .drop_nulls("pts_ppr")
    )

    actuals.write_parquet(DATA / "actuals_weekly.parquet")
    summary = actuals.group_by("season").agg(
        rows=pl.len(), last_week=pl.col("week").max()
    ).sort("season")
    print(f"Wrote {actuals.height} player-games -> data/actuals_weekly.parquet")
    print(summary)
    print(f"(dropped {stats.height - actuals.height} rows with no Sleeper id)")


if __name__ == "__main__":
    main()
