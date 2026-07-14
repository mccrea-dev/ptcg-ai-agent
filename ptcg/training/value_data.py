"""Build the value-net dataset: (board features -> did the actor win?).

Two sources, per Gabe's call (2026-07-14):
- Mined ladder decisions (decisions_v0.jsonl): meta-wide expert judgment.
- Local self-play (tactical vs tactical): dense coverage of our own deck.

Positions within one game share an outcome, so we sample a handful per
(episode, actor) to decorrelate; labels are the final result, optionally
blended toward late-game certainty; samples are weighted by actor rating so
1300-team judgments count more than 600-team coin flips.

Usage:
  uv run python -m ptcg.training.value_data mined data/train/decisions_v0.jsonl -o data/train/value_mined.npz
  uv run python -m ptcg.training.value_data selfplay 4000 -o data/train/value_selfplay.npz
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from ptcg.training.encoding import board_features, build_card_vocab, board_card_ids

POSITIONS_PER_GAME = 8
CARD_LIMIT = 24


def _extra_features(board) -> list[float]:
    """Value-specific additions the BC feature set lacks: energy development,
    which for most decks is the tempo variable."""
    from ptcg.training.encoding import _get
    me = _get(board, "yourIndex")
    players = _get(board, "players")
    feats = []
    for p in (me, 1 - me):
        ps = players[p]
        pokes = [x for x in (_get(ps, "active") or []) + (_get(ps, "bench") or [])
                 if x is not None]
        feats.append(sum(len(_get(x, "energies") or []) for x in pokes) / 8.0)
        feats.append(sum(1 for x in pokes if len(_get(x, "energies") or []) >= 1) / 6.0)
    return feats


def encode_position(board, vocab) -> tuple[list[int], list[float]]:
    ids = board_card_ids(board, vocab, CARD_LIMIT)
    ids = ids + [0] * (CARD_LIMIT - len(ids))
    return ids, board_features(board) + _extra_features(board)


def _emit(records, vocab, rng):
    """records: iterable of (key, board, won, weight). Groups by key, samples
    POSITIONS_PER_GAME per group, encodes."""
    by_game = defaultdict(list)
    for key, board, won, weight in records:
        by_game[key].append((board, won, weight))
    ids_out, feat_out, y_out, w_out = [], [], [], []
    for _, positions in by_game.items():
        take = positions if len(positions) <= POSITIONS_PER_GAME \
            else rng.sample(positions, POSITIONS_PER_GAME)
        for board, won, weight in take:
            ids, feats = encode_position(board, vocab)
            ids_out.append(ids)
            feat_out.append(feats)
            y_out.append(1.0 if won else -1.0)
            w_out.append(weight)
    return (np.array(ids_out, dtype=np.int16), np.array(feat_out, dtype=np.float32),
            np.array(y_out, dtype=np.float32), np.array(w_out, dtype=np.float32))


def from_mined(path: Path, vocab, rng):
    def records():
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                # Rating-weighted: a 1300 team's positions are worth ~3x a
                # 700 team's; unrated actors count least.
                w = max(0.3, (r.get("rating") or 600.0) / 1000.0)
                yield (r["episode"], r["actor"]), r["board"], r["won"], w
    return _emit(records(), vocab, rng)


def from_selfplay(n_games: int, vocab, rng):
    import ptcg.engine.sdk  # noqa: F401
    from ptcg.agents import tactical
    from ptcg.decks import load_deck
    from ptcg.engine.fast import play_game

    deck = load_deck("meta/kadabra___alakazam")
    tac = tactical.make_agent(deck)
    out = []
    for g in range(n_games):
        seen = []

        def on_step(obs, action, seen=seen):
            cur = obs["current"]
            if cur.get("players"):
                seen.append((cur["yourIndex"], cur))

        winner = play_game(tac, tac, deck, deck, on_step=on_step)
        for actor, board in seen:
            out.append(((g, actor), board, winner == actor, 1.0))
        if (g + 1) % 500 == 0:
            print(f"  selfplay {g + 1}/{n_games}", flush=True)
    return _emit(out, vocab, rng)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["mined", "selfplay"])
    parser.add_argument("arg", help="jsonl path (mined) or game count (selfplay)")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    vocab = build_card_vocab()
    rng = random.Random(args.seed)
    if args.source == "mined":
        ids, feats, y, w = from_mined(Path(args.arg), vocab, rng)
    else:
        ids, feats, y, w = from_selfplay(int(args.arg), vocab, rng)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=ids, feats=feats, y=y, w=w)
    print(f"{len(y)} positions ({y.mean():+.3f} mean label) -> {args.out}")


if __name__ == "__main__":
    main()
