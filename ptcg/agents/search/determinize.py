"""Sampling of hidden zones for search.

The native search API needs concrete card IDs for everything we can't see:
our own deck order and prizes, and the opponent's deck, prizes, and hand.
We know our own 60 exactly, so our unseen pool is simple multiset arithmetic.
The opponent's 60 must be guessed; until we have meta data, callers pass a
guess (e.g. a mirror of our own deck).
"""

import random
from collections import Counter
from dataclasses import dataclass


@dataclass
class Determinization:
    your_deck: list[int]
    your_prize: list[int]
    opponent_deck: list[int]
    opponent_prize: list[int]
    opponent_hand: list[int]
    opponent_active: list[int]


def _visible_ids(state: dict, select: dict | None, p: int, include_hand: bool) -> list[int]:
    """Card IDs of player p's cards that have left their deck+prize.

    Keyed by each card's per-match serial so nothing is double-counted, and
    including the select's effect/contextCard: a card being played sits in no
    zone at all while its effect resolves."""
    seen: dict[int, int] = {}  # serial -> card id

    def add(card):
        if card is not None and card.get("playerIndex") == p:
            seen[card["serial"]] = card["id"]

    ps = state["players"][p]
    if include_hand and ps["hand"]:
        for c in ps["hand"]:
            add(c)
    for c in ps["discard"]:
        add(c)
    for zone in (ps["active"], ps["bench"]):
        for pkm in zone:
            if pkm is None:
                continue
            add(pkm)
            for key in ("energyCards", "tools", "preEvolution"):
                for c in pkm.get(key) or []:
                    add(c)
    for c in ps["prize"]:
        add(c)
    for c in state.get("stadium") or []:
        add(c)
    for c in state.get("looking") or []:
        add(c)
    if select is not None:
        # The effect card (a resolving Supporter/Item) sits in no zone at all.
        # contextCard must NOT be counted: for attach-from-deck effects it
        # refers to a card still inside deckCount, and subtracting it starves
        # the pool by one (531 occurrences in a 200-game audit).
        add(select.get("effect"))
    return list(seen.values())


def _unseen_pool(deck60: list[int], state: dict, select: dict | None,
                 p: int, include_hand: bool) -> list[int]:
    pool = Counter(deck60)
    pool.subtract(Counter(_visible_ids(state, select, p, include_hand)))
    if min(pool.values(), default=0) < 0:
        # Our accounting missed a zone (or the deck guess is wrong, which is
        # expected for opponents). Clamp rather than fail.
        pool = Counter({k: v for k, v in pool.items() if v > 0})
    return list(pool.elements())


def sample(obs: dict, my_deck60: list[int], opp_deck60_guess: list[int],
           rng: random.Random, my_facedown_active: int | None = None) -> Determinization:
    """Draw one consistent assignment of all hidden zones for the observation,
    from the perspective of the player to move.

    During the setup-bench window our own active is face-down and invisible
    even to us; callers that remember which card they chose pass it as
    `my_facedown_active` so the pool arithmetic still balances."""
    state = obs["current"]
    select = obs["select"]
    me = state["yourIndex"]
    opp = 1 - me

    mine = _unseen_pool(my_deck60, state, select, me, include_hand=True)
    my_active = state["players"][me]["active"]
    if my_active and my_active[0] is None:
        if my_facedown_active is None or my_facedown_active not in mine:
            raise ValueError("own active is face-down and unknown")
        mine.remove(my_facedown_active)
    theirs = _unseen_pool(opp_deck60_guess, state, select, opp, include_hand=False)
    rng.shuffle(mine)
    rng.shuffle(theirs)

    n_deck = state["players"][me]["deckCount"]
    n_prize = sum(1 for c in state["players"][me]["prize"] if c is None)
    o_deck = state["players"][opp]["deckCount"]
    o_prize = sum(1 for c in state["players"][opp]["prize"] if c is None)
    o_hand = state["players"][opp]["handCount"]

    # A face-down opponent active (setup phase) must be predicted explicitly.
    active = state["players"][opp]["active"]
    face_down = len(active) > 0 and active[0] is None
    opp_active = []
    if face_down:
        from ptcg.engine.sdk import api
        basics = {c.cardId for c in api.all_card_data() if c.basic}
        candidates = [cid for cid in theirs if cid in basics]
        if not candidates:
            raise ValueError("no basic Pokemon left in opponent deck guess")
        opp_active = [rng.choice(candidates)]
        theirs.remove(opp_active[0])

    return Determinization(
        your_deck=mine[:n_deck],
        your_prize=mine[n_deck:n_deck + n_prize],
        opponent_deck=theirs[:o_deck],
        opponent_prize=theirs[o_deck:o_deck + o_prize],
        opponent_hand=theirs[o_deck + o_prize:o_deck + o_prize + o_hand],
        opponent_active=opp_active,
    )
