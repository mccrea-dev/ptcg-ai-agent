"""Behavior-cloning baseline: a numpy softmax ranker over option features.

Learns to pick the option a high-rated ladder agent picked. Scores every
option with a shared linear function; softmax over the options in a selection;
cross-entropy on the chosen index. Reported against two baselines: random
guessing and agreement with our hand-written priority policy.

Usage: uv run python -m ptcg.training.train_bc data/train/decisions_v0.jsonl -o data/train/bc_v0.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ptcg.agents import priority_agent
from ptcg.training.features import N_FEATURES, featurize

MIN_RATING = 1000.0


def load_rows(path: Path):
    """Yield (episode, features[n_opt, F], target) for single-pick decisions
    made by high-rated actors."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["rating"] < MIN_RATING or len(r["chosen"]) != 1:
                continue
            select, board = r["select"], r["board"]
            options = select["option"] or []
            if len(options) < 2 or select.get("maxCount") != 1:
                continue
            feats = np.stack([featurize(o, select, board, i, len(options))
                              for i, o in enumerate(options)])
            yield r["episode"], feats, r["chosen"][0], select, options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.3)
    args = parser.parse_args()

    rows = list(load_rows(args.data))
    print(f"{len(rows)} usable decisions")
    episodes = sorted({ep for ep, *_ in rows})
    val_eps = set(episodes[:: 10])  # every 10th episode held out
    train = [r for r in rows if r[0] not in val_eps]
    val = [r for r in rows if r[0] in val_eps]
    print(f"train {len(train)}, val {len(val)} (episode-level split)")

    rng = np.random.default_rng(0)
    w = np.zeros(N_FEATURES, dtype=np.float32)
    for epoch in range(args.epochs):
        order = rng.permutation(len(train))
        loss_sum = 0.0
        for idx in order:
            _, feats, target, _, _ = train[idx]
            logits = feats @ w
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            loss_sum += -np.log(max(p[target], 1e-9))
            grad = feats.T @ p - feats[target]
            w -= args.lr * grad / len(feats)
        print(f"epoch {epoch}: train loss {loss_sum / len(train):.4f}")

    hits = prio_hits = rand_inv = 0
    for _, feats, target, select, options in val:
        hits += int(np.argmax(feats @ w) == target)
        prio = priority_agent.agent({"select": {"option": options, "maxCount": 1,
                                                "type": select.get("type"),
                                                "context": select.get("context")}})
        prio_hits += int(prio[0] == target)
        rand_inv += 1 / len(options)
    n = len(val)
    print(f"\nval top-1: model {hits / n:.1%} | priority {prio_hits / n:.1%} "
          f"| random {rand_inv / n:.1%}")

    np.savez(args.out, w=w, n_features=N_FEATURES)
    print("saved", args.out)


if __name__ == "__main__":
    main()
