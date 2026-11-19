"""
backtest.py

Scores projections against what actually happened this season, then fits the
in-season weekly model (see weekly_model.py) from the same data.

Projection methods compared, all using only information available before the
week was played:
    sleeper  Sleeper's final pregame projection
    ppg      the player's season points-per-game to date (Sleeper if no games yet)
    blend    credibility blend of the two -- our model
    espn     ESPN's pregame projection (rostered players only; ESPN doesn't
             give us the rest)

Honest evaluation: the blend's one knob per position (k) and the CV tiers are
fit on the early weeks and scored on the later weeks they never saw. The params
the daily pipeline actually uses are then refit on every finished week.

Accuracy is measured on games the player actually played -- whether he plays at
all is modeled separately, as a did-not-play rate.

Reads:  data/projections_weekly_all.parquet, data/actuals_weekly.parquet,
        data/league_weekly.parquet, data/schedule.parquet, data/playerids.parquet
Writes: data/weekly_params.parquet          (the fitted model)
        data/backtest.parquet               (holdout accuracy by method/position)
        data/backtest_weekly.parquet        (MAE by week, for the trend chart)
        data/backtest_calibration.parquet   (do our 80% ranges hold 80%?)
"""

import os

import numpy as np
import polars as pl
from dotenv import load_dotenv

from weekly_model import (DATA, OFFENSE, RATIO_GRID, TIER_FLOORS, blend_expr,
                          espn_to_sleeper, prior_production)

load_dotenv()
SEASON = int(os.environ.get("SEASON", 2026))
K_GRID = [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 1e9]   # 1e9 = "ignore ppg"
NO_BLEND = 1e9          # K and DEF: Sleeper only (no nflverse production for them)
MIN_TIER_N = 25         # a CV tier needs this many player-games or it merges down
DNP_MIN_PROJ = 3.0      # only count DNPs among players projected to matter
TALENT_MIN_GAMES = 4


# ---------------------------------------------------------------- data

def build_rows():
    """One row per projected player-week in finished weeks, with the actual
    result (null = didn't play), prior production, and ESPN's projection."""
    schedule_path = DATA / "schedule.parquet"
    if not schedule_path.exists():
        return None, None
    schedule = pl.read_parquet(schedule_path)
    last_done = schedule.filter(pl.col("outcome") != "U")["week"].max()
    if last_done is None:
        return None, None

    sleeper = (
        pl.read_parquet(DATA / "projections_weekly_all.parquet")
        .filter((pl.col("season") == SEASON) & (pl.col("week") <= last_done)
                & (pl.col("pts_ppr") > 0))
        .select("player_id", "name", "position", "week", pl.col("pts_ppr").alias("sleeper"))
    )
    actuals = pl.read_parquet(DATA / "actuals_weekly.parquet")
    prior = prior_production(actuals, SEASON)

    espn = espn_to_sleeper(
        pl.read_parquet(DATA / "league_weekly.parquet").filter(pl.col("week") <= last_done)
    )
    espn_proj = (
        espn.filter(pl.col("projected_points") > 0)
        .group_by("player_id", "week").agg(pl.col("projected_points").first().alias("espn"))
    )

    # Offense: truth is nflverse (every player, not just rostered ones).
    offense = (
        sleeper.filter(pl.col("position").is_in(OFFENSE))
        .join(actuals.filter(pl.col("season") == SEASON)
              .select("player_id", "week", pl.col("pts_ppr").alias("actual")),
              on=["player_id", "week"], how="left")
        .join(prior, on=["player_id", "week"], how="left")
    )

    # K / DEF: truth is ESPN's box score (rostered only), in league scoring.
    kdef_actual = (
        espn.filter(pl.col("position").is_in(["K", "DEF"]) & (pl.col("game_played") == 100))
        .group_by("player_id", "week").agg(pl.col("points").first().alias("actual"))
    )
    kdef = (
        sleeper.filter(pl.col("position").is_in(["K", "DEF"]))
        .join(kdef_actual, on=["player_id", "week"], how="inner")
        .with_columns(g_prior=pl.lit(None, pl.Int64), ppg_prior=pl.lit(None, pl.Float64))
    )

    rows = pl.concat([offense, kdef], how="diagonal_relaxed").join(
        espn_proj, on=["player_id", "week"], how="left")
    return rows, last_done


# ---------------------------------------------------------------- fitting

def metrics(df, pred):
    err = df["actual"] - df[pred]
    return {"n": df.height, "mae": float(err.abs().mean()),
            "rmse": float((err ** 2).mean() ** 0.5), "bias": float(err.mean())}


def scored(rows):
    """The player-weeks we score accuracy on, i.e. games he played.

    nflverse has no row for a player who was active but recorded nothing, so a
    fringe guy's zero-point weeks would vanish and his tier would look great.
    Below DNP_MIN_PROJ a missing row is far likelier "active, did nothing" than
    "inactive", so it counts as a 0. (The DNP rate is measured only at or above
    that line, so nothing is double-counted.)"""
    return rows.with_columns(
        actual=pl.when(pl.col("actual").is_null() & pl.col("position").is_in(OFFENSE)
                       & (pl.col("sleeper") < DNP_MIN_PROJ))
        .then(0.0).otherwise(pl.col("actual"))
    ).filter(pl.col("actual").is_not_null())


def fit(rows):
    """Fit k, CV tiers, DNP rate and talent spread per position."""
    played = scored(rows)
    records = []
    for pos in TIER_FLOORS:
        sub = played.filter(pl.col("position") == pos)
        if not sub.height:
            continue

        # k: grid search on MAE (offense only).
        if pos in OFFENSE:
            scores = {k: metrics(sub.with_columns(p=blend_expr(k)), "p")["mae"] for k in K_GRID}
            k = min(scores, key=scores.get)
        else:
            k = NO_BLEND
        sub = sub.with_columns(proj=blend_expr(k)).filter(pl.col("proj") > 0)

        # DNP rate among players projected to matter.
        if pos in OFFENSE:
            proj_all = rows.filter((pl.col("position") == pos) & (pl.col("sleeper") >= DNP_MIN_PROJ))
            dnp = proj_all["actual"].is_null().mean()
        else:
            dnp = 0.0

        # Talent spread: how much a player's average actual/projected ratio varies
        # across players beyond what week-to-week noise alone would produce. This
        # is projection error that PERSISTS -- it widens rest-of-season ranges.
        ratio = sub.filter(pl.col("proj") >= 5).with_columns(r=pl.col("actual") / pl.col("proj"))
        per = ratio.group_by("player_id").agg(
            n=pl.len(), m=pl.col("r").mean(), v=pl.col("r").var()
        ).filter(pl.col("n") >= TALENT_MIN_GAMES)
        talent_var = (per["m"].var() - (per["v"] / per["n"]).mean()) if per.height > 5 else 0.0
        talent_sd = float(np.sqrt(max(talent_var or 0.0, 0.0)))

        # CV tiers: merge any thin tier down into the one below it.
        floors = TIER_FLOORS[pos]
        kept = []
        for i, lo in enumerate(floors):
            hi = floors[i + 1] if i + 1 < len(floors) else float("inf")
            kept.append([lo, hi])
        tiers = []
        for lo, hi in kept:
            t = sub.filter((pl.col("proj") >= lo) & (pl.col("proj") < hi))
            if t.height < MIN_TIER_N and tiers:
                tiers[-1][1] = hi           # extend the tier below
                continue
            tiers.append([lo, hi])
        for lo, hi in tiers:
            t = sub.filter((pl.col("proj") >= lo) & (pl.col("proj") < hi))
            if not t.height:
                continue
            rmse = float(((t["actual"] - t["proj"]) ** 2).mean() ** 0.5)
            ratio = (t["actual"] / t["proj"]).to_numpy()
            records.append({
                "position": pos, "proj_floor": float(lo), "n": t.height,
                "cv": rmse / float(t["proj"].mean()),
                # The weekly outcome distribution, as multiples of the projection.
                "ratio_q": [round(float(q), 4) for q in np.percentile(ratio, RATIO_GRID)],
                "mean_ratio": float(ratio.mean()),
                "k": float(k), "dnp_rate": float(dnp), "talent_sd": talent_sd,
            })
    return pl.DataFrame(records)


def with_model(rows, params):
    """Attach the model's projection (blend) and its tier's ratio quantiles."""
    k = params.group_by("position").agg(pl.col("k").first())
    out = rows.join(k, on="position", how="inner").with_columns(proj=blend_expr(pl.col("k")))
    tiers = params.select("position", "proj_floor", "ratio_q").sort("proj_floor")
    return out.sort("position", "proj").join_asof(
        tiers, left_on="proj", right_on="proj_floor", by="position", strategy="backward"
    )


# ---------------------------------------------------------------- evaluate

def evaluate(rows, params, test_weeks):
    test = with_model(scored(rows.filter(pl.col("week").is_in(test_weeks))), params) \
        .with_columns(ppg=pl.coalesce("ppg_prior", "sleeper"))

    out = []
    for pos in ["ALL", *TIER_FLOORS]:
        sub = test if pos == "ALL" else test.filter(pl.col("position") == pos)
        if not sub.height:
            continue
        for method, col in [("sleeper", "sleeper"), ("ppg", "ppg"), ("blend", "proj")]:
            out.append({"subset": "All projected players", "position": pos,
                        "method": method, **metrics(sub, col)})
        rost = sub.drop_nulls("espn")
        for method, col in [("sleeper", "sleeper"), ("blend", "proj"), ("espn", "espn")]:
            if rost.height:
                out.append({"subset": "Rostered (ESPN available)", "position": pos,
                            "method": method, **metrics(rost, col)})
    return pl.DataFrame(out).with_columns(test_weeks=pl.lit(f"{min(test_weeks)}-{max(test_weeks)}"))


def calibration(rows, params, test_weeks):
    """Share of actual results inside our p10-p90 range. 80% = honest ranges;
    lower = overconfident, higher = too timid."""
    test = with_model(scored(rows.filter(pl.col("week").is_in(test_weeks))), params) \
        .filter((pl.col("proj") > 0) & pl.col("ratio_q").is_not_null())
    # The range scales with the projection, so p10 / p90 are just the tier's
    # p10 / p90 ratio times the projection.
    lo = pl.col("proj") * pl.col("ratio_q").list.get(RATIO_GRID.index(10))
    hi = pl.col("proj") * pl.col("ratio_q").list.get(RATIO_GRID.index(90))
    test = test.with_columns(
        inside=(pl.col("actual") >= lo) & (pl.col("actual") <= hi),
        below=pl.col("actual") < lo, above=pl.col("actual") > hi,
    )
    by = lambda d: d.agg(n=pl.len(), coverage=pl.col("inside").mean(),
                         below_floor=pl.col("below").mean(), above_ceiling=pl.col("above").mean())
    return pl.concat([
        by(test.group_by("position")),
        by(test.select(pl.lit("ALL").alias("position"), "inside", "below", "above").group_by("position")),
    ])


def weekly_trend(rows, params):
    df = with_model(scored(rows), params)
    return (
        df.group_by("week").agg(
            sleeper=(pl.col("actual") - pl.col("sleeper")).abs().mean(),
            blend=(pl.col("actual") - pl.col("proj")).abs().mean(),
            n=pl.len(),
        )
        .unpivot(index=["week", "n"], on=["sleeper", "blend"], variable_name="method",
                 value_name="mae")
        .sort("week", "method")
    )


def main():
    rows, last_done = build_rows()
    if rows is None or rows["week"].n_unique() < 2:
        # Preseason / week 1: nothing to score yet. Keep last season's params
        # (if any) so the in-season script still has something to run on.
        print("Fewer than two finished weeks -- skipping the backtest for now.")
        return
    weeks = sorted(rows["week"].unique().to_list())
    print(f"{rows.height} projected player-weeks across weeks {weeks[0]}-{weeks[-1]}")

    # Hold out roughly the last 40% of finished weeks.
    split = max(2, int(len(weeks) * 0.6))
    train_w, test_w = weeks[:split], weeks[split:]

    if test_w:
        holdout_params = fit(rows.filter(pl.col("week").is_in(train_w)))
        bt = evaluate(rows, holdout_params, test_w)
        cal = calibration(rows, holdout_params, test_w)
        trend = weekly_trend(rows, holdout_params).with_columns(
            holdout=pl.col("week").is_in(test_w))
        bt.write_parquet(DATA / "backtest.parquet")
        cal.write_parquet(DATA / "backtest_calibration.parquet")
        trend.write_parquet(DATA / "backtest_weekly.parquet")
        print(f"\nHoldout (fit weeks {train_w[0]}-{train_w[-1]}, scored weeks "
              f"{test_w[0]}-{test_w[-1]}):")
        print(bt.filter(pl.col("subset") == "All projected players")
              .pivot(on="method", index="position", values="mae"))
        print(bt.filter(pl.col("subset") != "All projected players")
              .pivot(on="method", index="position", values="mae"))
        print(cal.sort("position"))

    params = fit(rows)
    params.write_parquet(DATA / "weekly_params.parquet")
    print("\nProduction params (all finished weeks) -> data/weekly_params.parquet")
    print(params)


if __name__ == "__main__":
    main()
