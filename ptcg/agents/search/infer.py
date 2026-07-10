"""Guess the opponent's deck from what they've revealed.

Every card the opponent has shown (board, discard, stadium...) constrains
which known meta deck they could be playing. We pick the library deck that
explains their revealed cards best; if nothing fits (an off-meta deck), the
caller's fallback guess is used. The library CSVs come from ladder episode
mining (ptcg.data.scout --export-decks).
"""

from collections import Counter
from pathlib import Path

from ptcg.agents.search.determinize import _visible_ids

_META_DIR = Path(__file__).parents[2] / "decks" / "meta"
_MAX_UNEXPLAINED = 5  # revealed copies a candidate may fail to explain

_library: list[tuple[str, list[int], Counter]] | None = None


def library() -> list[tuple[str, list[int], Counter]]:
    global _library
    if _library is None:
        _library = []
        for path in sorted(_META_DIR.glob("*.csv")):
            deck = [int(x) for x in path.read_text().splitlines() if x.strip()]
            _library.append((path.stem, deck, Counter(deck)))
    return _library


def guess_deck(obs: dict, fallback: list[int]) -> list[int]:
    """Best-fitting known deck for the opponent, or `fallback` if none fits."""
    state = obs["current"]
    opp = 1 - state["yourIndex"]
    revealed = Counter(_visible_ids(state, obs["select"], opp, include_hand=False))

    best_deck, best_key = None, None
    for _, deck, counts in library():
        unexplained = sum((revealed - counts).values())
        overlap = sum((revealed & counts).values())
        key = (unexplained, -overlap)
        if best_key is None or key < best_key:
            best_key, best_deck = key, deck
    if best_deck is None or best_key[0] > _MAX_UNEXPLAINED:
        return fallback
    return best_deck
