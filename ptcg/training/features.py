"""Feature vectors for scoring one option within a selection.

Used identically at training time (on replay dicts) and at inference (on the
search API's dataclass objects), so every accessor tolerates both shapes.
The vector is deliberately small and hand-engineered: this is the numpy
baseline model; a learned encoder can replace it later without touching the
training pipeline's structure.
"""

import numpy as np

N_OPTION_TYPES = 17
N_SELECT_TYPES = 11
N_CARD_TYPES = 7

# option-type one-hot + select-type one-hot + card features + scalars
N_FEATURES = N_OPTION_TYPES + N_SELECT_TYPES + N_CARD_TYPES + 14

_card_index = None
_attack_damage = None


def _cards():
    global _card_index
    if _card_index is None:
        from ptcg.engine.sdk import api
        _card_index = {c.cardId: c for c in api.all_card_data()}
    return _card_index


def _attacks():
    global _attack_damage
    if _attack_damage is None:
        from ptcg.agents.priority_agent import ATTACK_DAMAGE
        _attack_damage = ATTACK_DAMAGE
    return _attack_damage


def _get(x, key):
    if isinstance(x, dict):
        return x.get(key)
    return getattr(x, key, None)


def featurize(option, select, board, position: int, n_options: int) -> np.ndarray:
    """One option in context -> feature vector."""
    f = np.zeros(N_FEATURES, dtype=np.float32)
    opt_type = _get(option, "type") or 0
    if isinstance(opt_type, str):  # replay dicts store enum names sometimes
        opt_type = 0
    f[min(int(opt_type), N_OPTION_TYPES - 1)] = 1.0

    sel_type = _get(select, "type") or 0
    if not isinstance(sel_type, str):
        f[N_OPTION_TYPES + min(int(sel_type), N_SELECT_TYPES - 1)] = 1.0

    base = N_OPTION_TYPES + N_SELECT_TYPES
    card_id = _get(option, "cardId")
    card = _cards().get(card_id) if card_id else None
    if card is not None:
        f[base + min(int(card.cardType), N_CARD_TYPES - 1)] = 1.0
    base += N_CARD_TYPES

    # scalar block
    if card is not None:
        f[base + 0] = (card.hp or 0) / 300.0
        f[base + 1] = float(card.basic)
        f[base + 2] = float(card.stage1 or card.stage2)
        f[base + 3] = float(card.ex)
        f[base + 4] = float(card.megaEx)
    attack_id = _get(option, "attackId")
    if attack_id:
        f[base + 5] = 1.0
        f[base + 6] = _attacks().get(attack_id, 0) / 300.0
    f[base + 7] = position / max(n_options, 1)
    f[base + 8] = min(n_options, 20) / 20.0
    if board is not None:
        me = _get(board, "yourIndex") or 0
        players = _get(board, "players") or [None, None]
        f[base + 9] = (_get(board, "turn") or 0) / 30.0
        mine, theirs = players[me], players[1 - me]
        if mine is not None:
            prize = _get(mine, "prize") or []
            f[base + 10] = sum(1 for c in prize if c is None) / 6.0
            f[base + 11] = (_get(mine, "handCount") or 0) / 10.0
        if theirs is not None:
            prize = _get(theirs, "prize") or []
            f[base + 12] = sum(1 for c in prize if c is None) / 6.0
    f[base + 13] = 1.0  # bias
    return f
