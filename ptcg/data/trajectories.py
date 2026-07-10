"""Extract (observation, action) training decisions from ladder replays.

Replays embed a `visualize` array with, per decision, the acting player's
actual observation (`obs`) and their chosen option indices (`selected`) —
exactly the supervised pairs imitation learning needs. We keep the actor's
observation (minus logs, which dominate size), the choice, and outcome/rating
metadata for filtering and weighting at training time.

Usage: uv run python -m ptcg.data.trajectories data/episodes/top_teams -o data/train/decisions.jsonl
"""

import argparse
import json
from pathlib import Path

from ptcg.data.scout import load_leaderboard


def extract_episode(path: Path, ratings: dict[str, float]):
    ep = json.loads(path.read_text(encoding="utf-8"))
    vis = (ep.get("steps") or [[{}]])[0][0].get("visualize")
    if not vis:
        return
    teams = ep.get("info", {}).get("TeamNames", ["?", "?"])
    rewards = [ep["steps"][-1][i].get("reward") for i in (0, 1)]
    episode_id = ep.get("id", 0)
    for entry in vis:
        obs = entry.get("obs") or {}
        select = obs.get("select")
        cur = obs.get("current")
        chosen = entry.get("selected")
        if not select or not cur or chosen is None:
            continue  # deck selection or malformed entry
        actor = cur["yourIndex"]
        yield {
            "episode": episode_id,
            "actor": teams[actor],
            "rating": ratings.get(teams[actor], 0.0),
            "won": rewards[actor] == 1,
            "turn": cur["turn"],
            "select": {k: select.get(k) for k in
                       ("type", "context", "minCount", "maxCount", "option")},
            "board": cur,
            "chosen": chosen,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_dirs", type=Path, nargs="+")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()

    ratings = load_leaderboard()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_records = n_episodes = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for d in args.replay_dirs:
            for path in sorted(d.glob("*.json")):
                wrote = False
                for record in extract_episode(path, ratings):
                    f.write(json.dumps(record) + "\n")
                    n_records += 1
                    wrote = True
                n_episodes += wrote
    print(f"{n_records} decisions from {n_episodes} episodes -> {args.out}")


if __name__ == "__main__":
    main()
