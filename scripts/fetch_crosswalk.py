"""
fetch_crosswalk.py

Downloads the nflverse player-ID crosswalk and saves a slim copy to
data/playerids.parquet.

Why this script exists:
Sleeper's projections are keyed by Sleeper's own player IDs, and Sleeper's
documentation for "which ID is which player" is un>reliable. Instead of trying
to decode that ourselves, we borrow nflverse's community-maintained crosswalk,
which already maps a single player across every major platform's ID system
(Sleeper, ESPN, and many more) plus their real name, team, and position.

Every other script joins to this file, so it runs first.
"""
import sys
print(sys.executable)

from pathlib import Path

import nflreadpy as nfl
import polars as pl

# data/ is our "handoff shelf" -- the folder the app reads from.
# mkdir(exist_ok=True) creates it if it's missing and does nothing if it's there.
DATA = Path(__file__).parent.parent / "data" 
# Path(__file__) is the path to this script.
# Setting that the file goes there, not to WD
DATA.mkdir(exist_ok=True)

# One row per player, with an ID column for each platform. Returns a Polars
# DataFrame (nflreadpy uses Polars natively, which is what we want).
ids = nfl.load_ff_playerids()

# We only need a handful of columns for our joins. We intersect our wish list
# with what actually came back, so a renamed/missing column won't crash the
# script -- it just gets skipped.
wanted = ["sleeper_id", "espn_id", "name", "merge_name", "position", "team"]
keep = [c for c in wanted if c in ids.columns]
crosswalk = ids.select(keep)

# Projections are keyed by sleeper_id, so a player with no Sleeper ID is useless
# for our join. Drop those rows to keep the file small and clean.
# pl.col("...") is how Polars refers to a column inside an operation -- you'll
# use this constantly.
if "sleeper_id" in crosswalk.columns:
    crosswalk = crosswalk.filter(pl.col("sleeper_id").is_not_null())

crosswalk.write_parquet(DATA / "playerids.parquet")

# Feedback so that when you run this later you can see it worked and eyeball
# the ID columns (their exact data types matter for joins -- more on that when
# we write the projections script).
print(f"Wrote {crosswalk.height} players to data/playerids.parquet")
print("Columns:", crosswalk.columns)
print(crosswalk.head())

