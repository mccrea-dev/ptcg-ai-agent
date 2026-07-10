"""Torch-free encoding shared by the trainer (torch) and bundle inference
(numpy): board feature extraction, card-id pooling, and the dimension
constants both sides must agree on."""

N_OPTION_TYPES = 17
N_SELECT_TYPES = 11
N_CONTEXTS = 50
N_BOARD_FEATS = 12
PAD = 0  # card index 0 reserved for "no card"


def _get(x, key):
    if isinstance(x, dict):
        return x.get(key)
    return getattr(x, key, None)


def build_card_vocab() -> dict[int, int]:
    """cardId -> contiguous embedding index (0 reserved)."""
    from ptcg.engine.sdk import api
    return {c.cardId: i + 1 for i, c in enumerate(api.all_card_data())}


def board_features(board) -> list[float]:
    me = _get(board, "yourIndex")
    players = _get(board, "players")
    feats = [(_get(board, "turn") or 0) / 30.0]
    for p in (me, 1 - me):
        ps = players[p]
        active_zone = _get(ps, "active") or []
        active = active_zone[0] if active_zone else None
        hp = _get(active, "hp") if active else None
        max_hp = _get(active, "maxHp") if active else None
        feats += [
            sum(1 for c in (_get(ps, "prize") or []) if c is None) / 6.0,
            (_get(ps, "handCount") or 0) / 10.0,
            (_get(ps, "deckCount") or 0) / 60.0,
            (hp / max_hp) if hp and max_hp else 0.0,
            len(_get(ps, "bench") or []) / 5.0,
        ]
    feats.append(1.0)
    return feats


def board_card_ids(board, vocab: dict[int, int], limit: int = 24) -> list[int]:
    """Card-embedding indices for pooling: our hand + both sides' Pokemon."""
    me = _get(board, "yourIndex")
    players = _get(board, "players")
    ids = []
    for c in _get(players[me], "hand") or []:
        ids.append(_get(c, "id"))
    for p in (me, 1 - me):
        ps = players[p]
        for zone in (_get(ps, "active") or [], _get(ps, "bench") or []):
            for pkm in zone:
                if pkm is not None:
                    ids.append(_get(pkm, "id"))
    return [vocab.get(i, PAD) for i in ids[:limit]] or [PAD]
