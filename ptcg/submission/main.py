"""Kaggle submission entry point. This file is bundled to the archive root;
Kaggle unpacks it at /kaggle_simulations/agent/ and calls agent() directly.

Clock-aware Monte Carlo search (ptcg.agents.search.mc_agent) with the
tactical policy as its internal fallback, and random-legal as the last-ditch
fallback if the ptcg package itself fails to load. A crash or timeout is an
automatic loss, so every layer degrades instead of raising.
"""

import os
import random
import sys

# kaggle_environments exec()s this file, so __file__ does not exist. The
# bundle root is either the working directory (local verify) or the fixed
# Kaggle agent path; make both importable.
for _p in (os.getcwd(), "/kaggle_simulations/agent"):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def read_deck_csv() -> list[int]:
    path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as f:
        return [int(line) for line in f.read().splitlines() if line.strip()]


_DECK = read_deck_csv()


def _find_value_net() -> str | None:
    for p in ("value_net.npz", "/kaggle_simulations/agent/value_net.npz"):
        if os.path.exists(p):
            return p
    return None


try:
    from ptcg.agents.search.mc_agent import make_agent
    try:
        _search = make_agent(deck=_DECK, value_path=_find_value_net())
    except Exception:
        _search = make_agent(deck=_DECK)  # net failed to load: hand eval
except Exception:
    _search = None


def agent(obs_dict: dict) -> list[int]:
    if obs_dict["select"] is None:
        return _DECK
    if _search is not None:
        try:
            return _search(obs_dict)
        except Exception:
            pass
    select = obs_dict["select"]
    k = min(select["maxCount"], len(select["option"]))
    return random.sample(range(len(select["option"])), k)
