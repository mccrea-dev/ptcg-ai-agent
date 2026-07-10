"""Ladder scouting: parse downloaded episode JSONs into decks/teams/results
and summarize the metagame, weighted by leaderboard rating.

Episodes come from the daily public datasets
(kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD); the leaderboard CSV comes
from `kaggle competitions leaderboard --download`. Team names join the two.

Usage: uv run python -m ptcg.data.scout data/episodes/2026-07-08
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ptcg.engine.sdk import api

EPISODES_ROOT = Path(__file__).parents[2] / "data" / "episodes"


@dataclass
class Episode:
    episode_id: int
    teams: list[str]
    decks: list[list[int] | None]
    winner: int  # 0 / 1 / -1 draw


def parse_episode(path: Path) -> Episode:
    ep = json.loads(path.read_text(encoding="utf-8"))
    steps = ep["steps"]
    decks: list[list[int] | None] = [None, None]
    for step in steps[:3]:
        for i, agent_state in enumerate(step):
            action = agent_state.get("action")
            if isinstance(action, list) and len(action) == 60 and decks[i] is None:
                decks[i] = action
    rewards = [steps[-1][0].get("reward"), steps[-1][1].get("reward")]
    winner = 0 if rewards[0] == 1 else 1 if rewards[1] == 1 else -1
    return Episode(
        episode_id=ep.get("id", 0),
        teams=ep.get("info", {}).get("TeamNames", ["?", "?"]),
        decks=decks,
        winner=winner,
    )


def archetype(deck: list[int]) -> str:
    """Label a deck by its two most-included non-basic or ex Pokemon."""
    cards = _card_index()
    key = Counter(
        cid for cid in deck
        if cid in cards and cards[cid].cardType == 0 and (not cards[cid].basic or cards[cid].ex)
    )
    if not key:
        key = Counter(cid for cid in deck if cid in cards and cards[cid].cardType == 0)
    return " / ".join(cards[c].name for c, _ in key.most_common(2)) or "unknown"


_cards = None


def _card_index():
    global _cards
    if _cards is None:
        _cards = {c.cardId: c for c in api.all_card_data()}
    return _cards


def load_leaderboard() -> dict[str, float]:
    """team name -> rating, from the newest downloaded leaderboard CSV."""
    files = sorted(EPISODES_ROOT.glob("pokemon-tcg-ai-battle-publicleaderboard-*.csv"))
    if not files:
        return {}
    ratings = {}
    with open(files[-1], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ratings[row["TeamName"]] = float(row["Score"])
    return ratings


def report(episode_dir: Path) -> None:
    ratings = load_leaderboard()
    episodes = [parse_episode(p) for p in sorted(episode_dir.glob("*.json"))]
    print(f"{len(episodes)} episodes, leaderboard teams known: {len(ratings)}")

    seat_count = Counter()
    wins = Counter()
    best_rating: dict[str, float] = defaultdict(float)
    for ep in episodes:
        for i in (0, 1):
            if ep.decks[i] is None:
                continue
            arch = archetype(ep.decks[i])
            seat_count[arch] += 1
            if ep.winner == i:
                wins[arch] += 1
            team = ep.teams[i]
            if team in ratings:
                best_rating[arch] = max(best_rating[arch], ratings[team])

    print(f"\n{'archetype':<45} {'seats':>5} {'winrate':>8} {'best rating seen':>17}")
    for arch, n in seat_count.most_common(20):
        wr = wins[arch] / n if n else 0
        print(f"{arch:<45} {n:>5} {wr:>7.0%} {best_rating[arch]:>17.0f}")


def export_decks(episode_dir: Path, out_dir: Path, top_n: int = 8) -> None:
    """Write the most common archetypes' deck lists as CSVs, picking the
    highest-rated team's exact 60 within each archetype. These both seed the
    eval gauntlet and give the determinizer candidate opponent decks."""
    ratings = load_leaderboard()
    by_arch: dict[str, list[tuple[float, list[int]]]] = defaultdict(list)
    for path in sorted(episode_dir.glob("*.json")):
        ep = parse_episode(path)
        for i in (0, 1):
            if ep.decks[i] is None:
                continue
            rating = ratings.get(ep.teams[i], 0.0)
            by_arch[archetype(ep.decks[i])].append((rating, ep.decks[i]))

    out_dir.mkdir(parents=True, exist_ok=True)
    popular = sorted(by_arch.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_n]
    for arch, entries in popular:
        rating, deck = max(entries, key=lambda e: e[0])
        slug = "".join(ch if ch.isalnum() else "_" for ch in arch.lower()).strip("_")
        path = out_dir / f"{slug}.csv"
        path.write_text("\n".join(str(c) for c in deck) + "\n")
        print(f"{path.name}: {len(entries)} seats seen, best rating {rating:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--export-decks", type=Path, default=None,
                        help="directory to write top archetype deck CSVs")
    args = parser.parse_args()
    if args.export_decks:
        export_decks(args.episode_dir, args.export_decks)
    else:
        report(args.episode_dir)
