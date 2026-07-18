"""
generate_recap.py

Turns a completed week's results into a funny message-board recap and writes it
to data/recap.md, which the dashboard renders on the League tab.

How it fits the pipeline: this runs in the WEEKLY job (Tuesday, after games are
final), not the daily one. A recap summarizes a finished week, so it's generated
once and then frozen until the next Tuesday -- regenerating it daily would burn
tokens and make the text wobble around for no reason.

Offseason note: with no games played yet, there's nothing real to recap. So if
no completed week exists, the script runs in DEMO mode -- it invents sample
scores for your real teams so you can test the whole loop (API call -> recap ->
dashboard) today. In-season it uses real results automatically, no code change.

Reads:  data/matchups.parquet, data/standings.parquet, data/transactions.parquet
Writes: data/recap.md
"""

import os
from pathlib import Path

import anthropic
import polars as pl
from dotenv import load_dotenv

load_dotenv()  # pulls ANTHROPIC_API_KEY (and the rest) from .env locally

DATA = Path(__file__).parent.parent / "data"

# Sonnet 5 is a good balance of humor quality and cost for a weekly creative
# task. To trim cost further you can swap in "claude-haiku-4-5-20251001".
MODEL = "claude-sonnet-5"

SYSTEM = (
    "You are the commissioner of a 12-team fantasy football league writing the "
    "weekly recap for the league message board. Your voice is funny, very  "
    "roast-y, and affectionate -- like a guy's group chat, not a newspaper. Write two "
    "or three short, punchy paragraphs. Use the actual team names and numbers "
    "you are given: hype the blowout winner, gently mock the lowest scorer, and "
    "play up the closest game. Keep it rated R. Do not invent facts, players, or "
    "scores beyond what you are given. Output plain markdown with no headers."
)


def load(name):
    path = DATA / f"{name}.parquet"
    return pl.read_parquet(path) if path.exists() else None


def week_facts(matchups, transactions):
    """Turn the week's tables into a compact list of plain-language facts.

    We do the counting here in Python -- who won, by how much, who was high and
    low -- and hand the model clean facts rather than raw tables. That keeps it
    from having to do arithmetic (which it can get wrong) and keeps it honest."""
    scored = []   # (team, points) across every team, for high/low
    margins = []  # (margin, winner, winner_pts, loser, loser_pts)
    score_lines = []

    for g in matchups.to_dicts():
        h, hs = g["home_team"], g["home_score"]
        a, as_ = g["away_team"], g["away_score"]
        if hs is None or as_ is None:   # skip byes / unplayed
            continue
        scored += [(h, hs), (a, as_)]
        if hs >= as_:
            margins.append((hs - as_, h, hs, a, as_))
        else:
            margins.append((as_ - hs, a, as_, h, hs))
        score_lines.append(f"{h} {hs:.1f} vs {a} {as_:.1f}")

    top = max(scored, key=lambda x: x[1])
    low = min(scored, key=lambda x: x[1])
    blowout = max(margins, key=lambda x: x[0])
    nail = min(margins, key=lambda x: x[0])

    facts = ["Scores this week:"]
    facts += [f"  - {ln}" for ln in score_lines]
    facts.append(f"Highest scorer: {top[0]} ({top[1]:.1f})")
    facts.append(f"Lowest scorer: {low[0]} ({low[1]:.1f})")
    facts.append(f"Biggest blowout: {blowout[1]} beat {blowout[3]} by "
                 f"{blowout[0]:.1f} ({blowout[2]:.1f}-{blowout[4]:.1f})")
    facts.append(f"Closest game: {nail[1]} edged {nail[3]} by {nail[0]:.1f} "
                 f"({nail[2]:.1f}-{nail[4]:.1f})")

    if transactions is not None and transactions.height:
        facts.append("Recent roster moves:")
        for m in transactions.head(6).to_dicts():
            facts.append(f"  - {m['team']}: {m['action']} {m['player']}")

    return "\n".join(facts)


def demo_facts(standings):
    """Build sample facts from REAL team names so the recap can be tested before
    any games have been played. Scores are invented and clearly fake."""
    names = (standings["team_name"].to_list() if standings is not None
             else [f"Team {i}" for i in range(1, 7)])
    n = min(6, len(names) - len(names) % 2)  # even number, up to 6 teams
    demo_scores = [128.4, 96.2, 141.7, 139.9, 88.0, 152.6]
    rows = [{
        "home_team": names[i], "home_score": demo_scores[i],
        "away_team": names[i + 1], "away_score": demo_scores[i + 1],
    } for i in range(0, n, 2)]
    return week_facts(pl.DataFrame(rows), None)


def generate_recap(facts_text):
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    msg = client.messages.create(
        model=MODEL,
        max_tokens=700,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Here are this week's facts:\n\n{facts_text}\n\nWrite the recap.",
        }],
    )
    # A text response comes back as one or more content blocks; join their text.
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def main():
    matchups = load("matchups")
    standings = load("standings")
    transactions = load("transactions")

    if matchups is not None and matchups.height:
        facts = week_facts(matchups, transactions)
        demo = False
    else:
        facts = demo_facts(standings)
        demo = True
        print("No completed week found -- running in DEMO mode with sample "
              "scores so you can test the recap end to end.\n")

    print("Facts sent to the model:\n" + facts + "\n")

    recap = generate_recap(facts)
    if demo:
        recap = ("_Preview generated from sample data -- real recaps appear "
                 "during the season._\n\n") + recap

    (DATA / "recap.md").write_text(recap, encoding="utf-8")
    print("Wrote data/recap.md:\n")
    print(recap)


if __name__ == "__main__":
    main()