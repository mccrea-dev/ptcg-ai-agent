"""Baseline agent: plays the starter deck and picks uniformly random legal options.

Exists to (a) validate the whole pipeline end to end and (b) serve as the floor
every future agent must beat.
"""

import random

from ptcg.decks import load_deck

DECK = load_deck("starter")


def agent(obs: dict) -> list[int]:
    # The engine's first call has no options; it is asking for our deck.
    if obs["select"] is None:
        return DECK
    n_options = len(obs["select"]["option"])
    return random.sample(range(n_options), obs["select"]["maxCount"])
