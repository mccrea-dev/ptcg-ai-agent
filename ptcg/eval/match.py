"""Head-to-head evaluation between two agents.

Single games of PTCG are nearly meaningless (shuffle luck, coin flips), so
every comparison reports a win rate with a Wilson confidence interval. Seats
alternate each game to cancel first-player advantage.

Agent modules must expose agent(obs) -> list[int] and DECK (60 card IDs).

Usage:
    uv run python -m ptcg.eval.match ptcg.agents.priority_agent ptcg.agents.random_agent -n 400
--kaggle runs through the full kaggle-environments wrapper (~20x slower);
use it to double-check that results hold in the official harness.
"""

import argparse
import importlib
import math
import time
from dataclasses import dataclass


@dataclass
class MatchResult:
    wins: int  # wins for agent_a
    losses: int
    draws: int
    seconds: float

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    def wilson_interval(self, z: float = 1.96) -> tuple[float, float]:
        """95% confidence interval for the true win rate."""
        n = self.games
        if n == 0:
            return (0.0, 1.0)
        p = self.win_rate()
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return (center - margin, center + margin)


def run_match(agent_a, agent_b, deck_a, deck_b, n_games: int, use_kaggle: bool = False) -> MatchResult:
    if use_kaggle:
        from ptcg.engine.runner import play_game as kaggle_play

        def play(p0, p1, d0, d1):
            r = kaggle_play(p0, p1)
            return 0 if r.reward0 == 1 else 1 if r.reward1 == 1 else -1
    else:
        from ptcg.engine.fast import play_game as play

    wins = losses = draws = 0
    start = time.time()
    for i in range(n_games):
        # Alternate seats: agent_a plays first in even games.
        if i % 2 == 0:
            winner = play(agent_a, agent_b, deck_a, deck_b)
            won = winner == 0
        else:
            winner = play(agent_b, agent_a, deck_b, deck_a)
            won = winner == 1
        if winner == -1:
            draws += 1
        elif won:
            wins += 1
        else:
            losses += 1
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_games}: {wins}W-{losses}L-{draws}D", flush=True)
    return MatchResult(wins, losses, draws, time.time() - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_a", help="module exposing agent() and DECK, e.g. ptcg.agents.random_agent")
    parser.add_argument("agent_b")
    parser.add_argument("-n", "--n-games", type=int, default=100)
    parser.add_argument("--kaggle", action="store_true", help="run via kaggle-environments")
    args = parser.parse_args()

    mod_a = importlib.import_module(args.agent_a)
    mod_b = importlib.import_module(args.agent_b)
    r = run_match(mod_a.agent, mod_b.agent, mod_a.DECK, mod_b.DECK, args.n_games, args.kaggle)

    lo, hi = r.wilson_interval()
    print(f"{args.agent_a} vs {args.agent_b}: {r.wins}W-{r.losses}L-{r.draws}D")
    print(f"win rate {r.win_rate():.1%}, 95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"{r.seconds:.1f}s total, {r.seconds / max(r.games, 1) * 1000:.0f} ms/game")


if __name__ == "__main__":
    main()
