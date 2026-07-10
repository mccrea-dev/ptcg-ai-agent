"""Compact JSONL replays of local games.

One line per game: decks, winner, and every decision (turn, select type and
context, the chosen option dicts). Enough to narrate a game or hunt blunders
without storing full observations; full-state capture can come later if
training needs it.
"""

import json

from ptcg.engine.fast import play_game


def compact_step(obs: dict, action: list[int]) -> dict:
    cur, sel = obs["current"], obs["select"]
    return {
        "turn": cur["turn"],
        "player": cur["yourIndex"],
        "type": sel["type"],
        "context": sel["context"],
        "action": action,
        "chosen": [sel["option"][i] for i in action],
    }


def play_logged(policy0, policy1, deck0: list[int], deck1: list[int],
                path: str) -> int:
    """Play one fast-runner game, appending its replay line to `path`."""
    steps: list[dict] = []
    winner = play_game(policy0, policy1, deck0, deck1,
                       on_step=lambda obs, action: steps.append(compact_step(obs, action)))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"decks": [deck0, deck1], "winner": winner,
                            "steps": steps}) + "\n")
    return winner
