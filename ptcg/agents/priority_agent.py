"""Minimal priority policy: no card knowledge, just a sane ordering of option
types. This is the "does basic competence beat random?" experiment, and if it
wins, the seed of the search agent's rollout policy.

Turn logic in one line: develop the board (evolve, use abilities, play cards,
attach energy), then attack, and only end the turn when nothing else is left.
"""

import re

from ptcg.decks import load_deck

DECK = load_deck("starter")


def _attack_damage_estimates() -> dict[int, int]:
    """attackId -> estimated damage. Scaling attacks ("does X damage for each
    ...") report damage=0 in the metadata; estimate them at 3x the coefficient,
    which is roughly what a built deck achieves mid-game."""
    from ptcg.engine.sdk import api

    estimates = {}
    for a in api.all_attack():
        damage = a.damage
        if damage == 0 and a.text:
            m = re.search(r"does (\d+)(?:\+|\smore)? damage for each", a.text)
            if m:
                damage = int(m.group(1)) * 3
        estimates[a.attackId] = damage
    return estimates


try:
    ATTACK_DAMAGE = _attack_damage_estimates()
except ImportError:  # SDK not present; fall back to position-based attack choice
    ATTACK_DAMAGE = {}

# OptionType values from the official SDK (cg/api.py).
EVOLVE, ABILITY, PLAY, ATTACH, ATTACK = 9, 10, 7, 8, 13
RETREAT, DISCARD, END = 12, 11, 14
YES = 1

# Lower = chosen first. Attacking usually ends the turn, so it comes after
# all development options; retreating and discarding are near-never useful
# without a reason, so they rank below attack but above passing.
MAIN_PRIORITY = {EVOLVE: 0, ABILITY: 1, PLAY: 2, ATTACH: 3, ATTACK: 4,
                 RETREAT: 8, DISCARD: 9, END: 10}


_card_table = None


def _cards():
    global _card_table
    if _card_table is None:
        from ptcg.engine.sdk import api
        _card_table = {c.cardId: c for c in api.all_card_data()}
    return _card_table


def _get(x, key):
    if isinstance(x, dict):
        return x.get(key)
    return getattr(x, key, None)


def battle_ctx(current) -> dict | None:
    """Cheap per-decision context for attack evaluation: the opponent's active
    (weakness/resistance/hp/prize worth) and their prizes left. Tolerates both
    dict observations and search-API dataclasses."""
    if current is None:
        return None
    try:
        me = _get(current, "yourIndex")
        players = _get(current, "players")
        opp = players[1 - me]
        active_zone = _get(opp, "active")
        active = active_zone[0] if active_zone else None
        if active is None:
            return None
        data = _cards().get(_get(active, "id"))
        if data is None:
            return None
        return {
            "opp_hp": _get(active, "hp") or 0,
            "opp_weakness": data.weakness,
            "opp_resistance": data.resistance,
            "opp_prize_worth": 3 if data.megaEx else 2 if data.ex else 1,
            "opp_prizes_left": sum(1 for c in (_get(opp, "prize") or []) if c is None),
            "my_energy_type": _my_attack_type(players[me]),
        }
    except Exception:
        return None


def _my_attack_type(mine):
    active_zone = _get(mine, "active")
    active = active_zone[0] if active_zone else None
    if active is None:
        return None
    data = _cards().get(_get(active, "id"))
    return data.energyType if data else None


def effective_damage(attack_id: int, ctx: dict | None) -> int:
    """Damage estimate adjusted for the defender's weakness/resistance."""
    est = ATTACK_DAMAGE.get(attack_id, 0)
    if ctx is None:
        return est
    if ctx["opp_weakness"] is not None and ctx["opp_weakness"] == ctx["my_energy_type"]:
        est *= 2
    elif ctx["opp_resistance"] is not None and ctx["opp_resistance"] == ctx["my_energy_type"]:
        est = max(0, est - 30)
    return est


def _score(position: int, option: dict, use_damage: bool, ctx: dict | None = None) -> tuple:
    kind = option["type"] if isinstance(option, dict) else option.type
    attack_id = option.get("attackId") if isinstance(option, dict) else option.attackId
    if kind == ATTACK:
        if use_damage and attack_id in ATTACK_DAMAGE:
            eff = effective_damage(attack_id, ctx)
            # True lethal: this KO takes their last prizes — nothing else matters.
            if (ctx is not None and eff >= ctx["opp_hp"] > 0
                    and ctx["opp_prize_worth"] >= ctx["opp_prizes_left"]):
                return (-1,)
            return (MAIN_PRIORITY[ATTACK], -eff)
        # Attacks on a card are conventionally listed weakest first, so among
        # affordable attacks prefer the last one.
        return (MAIN_PRIORITY[ATTACK], -position)
    if kind == YES:
        return (0,)  # YES before NO: activate optional effects by default
    return (MAIN_PRIORITY.get(kind, 5), position)


def make_agent(use_damage: bool = True, use_ctx: bool = True):
    def agent(obs: dict) -> list[int]:
        if obs["select"] is None:
            return DECK
        select = obs["select"]
        options = select["option"]
        ctx = battle_ctx(obs.get("current")) if use_ctx else None
        ranked = sorted(range(len(options)),
                        key=lambda i: _score(i, options[i], use_damage, ctx))
        return ranked[: select["maxCount"]]

    return agent


agent = make_agent(use_damage=True)
