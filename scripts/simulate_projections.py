"""
simulate_projections.py

Turns each Sleeper SEASON projection into a range of outcomes -- floor (p10),
median (p50), ceiling (p90), plus a full percentile grid -- by simulating many
availability-gated, gamma-distributed seasons. Writes data/projections_sim.parquet.

The percentile grid (p05..p95 in 5-point steps) is what lets the dashboard draw
distribution curves without re-running the simulation. Each adjacent pair of
percentiles brackets 5% of the outcomes, so the app can reconstruct an
approximate density from them -- cheap to store, no modeling logic in the app.

v1 scope: this is an OUTCOME-variance model. It treats the Sleeper projection as
the correct mean and simulates weekly luck + injuries around it. It does NOT yet
model uncertainty in the projection itself (a WR40 who might really be a WR15) --
that's a planned refinement. The dashboard labels the range accordingly.

Reads:  data/variance_params.parquet     (tiered CV + availability; produced by
                                           the variance notebook and committed)
        data/projections_season.parquet  (Sleeper projections; refreshed daily)
Writes: data/projections_sim.parquet
"""

from pathlib import Path

import numpy as np
import polars as pl

DATA = Path(__file__).parent.parent / "data"

SEED = 42          # fixed so the daily run is reproducible
N_SIMS = 5000      # simulated seasons per player
N_WEEKS = 17       # games in a season (one bye)
POSITIONS = ["QB", "RB", "WR", "TE"]   # positions we have variance params for
PROJ_COL = "pts_ppr"                   # matches how variance was measured (PPR)

# Percentile grid saved per player. 5-point steps keep the file small while
# giving the dashboard enough resolution to draw a smooth-ish curve.
PCTS = list(range(5, 100, 5))          # 5, 10, ..., 95


def load_tiers(params):
    """Index tiers per position, sorted by floor descending, for fast lookup."""
    tiers = {}
    for pos in params["position"].unique().to_list():
        sub = params.filter(pl.col("position") == pos).sort("proj_floor", descending=True)
        tiers[pos] = list(zip(sub["proj_floor"].to_list(),
                              sub["cv"].to_list(),
                              sub["availability"].to_list()))
    return tiers


def classify(tiers, position, projection):
    """Pick the tier whose floor is the highest value still <= the projection.
    Returns (cv, availability), or None if the position isn't covered."""
    for floor, cv, avail in tiers.get(position, []):
        if projection >= floor:
            return cv, avail
    return None


def simulate_player(projection, cv, availability, rng, n_sims=N_SIMS, n_weeks=N_WEEKS):
    """Simulate n_sims seasons; return an array of season totals.

    mean_active divides by EXPECTED games (availability * weeks), not raw weeks,
    so zeroing out injured weeks leaves the season mean on the projection."""
    mean_active = projection / (availability * n_weeks)
    shape = 1.0 / (cv ** 2)          # gamma shape from CV
    scale = mean_active / shape       # = mean_active * cv**2
    weekly = rng.gamma(shape, scale, size=(n_sims, n_weeks))   # points if active
    active = rng.random((n_sims, n_weeks)) < availability      # availability gate
    return (weekly * active).sum(axis=1)


def main():
    params = pl.read_parquet(DATA / "variance_params.parquet")
    proj = pl.read_parquet(DATA / "projections_season.parquet")

    tiers = load_tiers(params)
    rng = np.random.default_rng(SEED)

    usable = proj.filter(pl.col("position").is_in(POSITIONS) & (pl.col(PROJ_COL) > 0))

    records = []
    for r in usable.iter_rows(named=True):
        hit = classify(tiers, r["position"], r[PROJ_COL])
        if hit is None:
            continue
        cv, avail = hit
        season = simulate_player(r[PROJ_COL], cv, avail, rng)
        qs = np.percentile(season, PCTS)

        rec = {"player_id": r["player_id"]}   # join key back to the projections table
        rec.update({f"p{p:02d}": round(float(v), 1) for p, v in zip(PCTS, qs)})
        # Named aliases for the table view (these are just p10 / p50 / p90).
        rec["sim_floor"] = rec["p10"]
        rec["sim_median"] = rec["p50"]
        rec["sim_ceiling"] = rec["p90"]
        records.append(rec)

    sim = pl.DataFrame(records)
    sim.write_parquet(DATA / "projections_sim.parquet")
    print(f"Simulated {sim.height} players -> data/projections_sim.parquet")
    print(sim.select(["player_id", "sim_floor", "sim_median", "sim_ceiling"]).head())


if __name__ == "__main__":
    main()