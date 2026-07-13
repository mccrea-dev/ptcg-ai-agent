"""Tactical policy for the Kadabra/Alakazam deck, built on the pixiux pattern.

Adopted from the ~800-rated public pixiux Lucario agent (data/kernels/pixiux):
every main-phase decision first builds a full (attacker, attack, target) plan —
gated to what is actually executable this turn (bench attackers need a retreat,
bench targets need Boss's Orders in hand, missing energy needs an energy card
in hand and no attach used) — and targets are chosen by prize economy. All
options then score on one descending scalar; attacking scores near-last so the
whole turn develops first, and trainers are individually gated (Boss's only
when the plan snipes, draw only when the deck can afford it).

Deck engine this encodes: Powerful Hand places 20 damage per card in our hand
(no weakness), the whole line attacks for 1 energy, and evolving draws cards
(Kadabra +2, Alakazam +3) — so hand size is damage, evolution is card
advantage, and every card played from hand costs 20 damage.

Used standalone as a baseline and as the search agent's rollout policy and
fallback. Stateless per decision so search rollouts can interleave games.
"""

from ptcg.agents.priority_agent import ATTACK_DAMAGE, _cards, _get
from ptcg.decks import load_deck

DECK = load_deck("meta/kadabra___alakazam")

# OptionType values from the official SDK (cg/api.py).
OPT_NUMBER, OPT_YES, OPT_NO, OPT_CARD = 0, 1, 2, 3
OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ABILITY = 7, 8, 9, 10
OPT_RETREAT, OPT_ATTACK, OPT_END = 12, 13, 14

# SelectContext values (cg/api.py).
CTX_MAIN, CTX_SETUP_ACTIVE, CTX_SETUP_BENCH, CTX_SWITCH, CTX_TO_ACTIVE = 0, 1, 2, 3, 4
CTX_TO_BENCH = 5
CTX_TO_HAND, CTX_DISCARD = 7, 8
CTX_TO_DECK, CTX_TO_DECK_BOTTOM = 9, 10
CTX_ATTACH_FROM = 21
CTX_IS_FIRST = 41

AREA_HAND, AREA_DISCARD, AREA_ACTIVE, AREA_BENCH, AREA_PRIZE = 2, 3, 4, 5, 6

# Our deck's cards.
ABRA, KADABRA, ALAKAZAM = 741, 742, 743
DUNSPARCE, DUDUNSPARCE = 305, 66
PSY_ENERGY, TELEPATH_ENERGY, ENRICHING_ENERGY = 5, 19, 13
DAWN, HILDA, POKE_PAD, POFFIN = 1231, 1225, 1152, 1086
ENH_HAMMER, RARE_CANDY, BOSS, NIGHT_STRETCHER = 1081, 1079, 1182, 1097
BATTLE_CAGE, LANAS_AID, SACRED_ASH = 1264, 1184, 1129
ENERGY_IDS = {PSY_ENERGY, TELEPATH_ENERGY, ENRICHING_ENERGY}
POWERFUL_HAND = 1072

LOW_DECK_COUNT = 10

# How much energy each of our Pokemon ever wants. Dudunsparce shuffles itself
# and everything attached back into the deck, so feeding it loses the energy.
ENERGY_QUOTA = {ABRA: 1, KADABRA: 1, ALAKAZAM: 1, DUNSPARCE: 1, DUDUNSPARCE: 0}

_attack_cost = None
_quota_cache: dict[int, int] = {}


def _quota(card_id: int) -> int:
    """Energy this Pokemon wants: our tuned table, else its priciest attack.
    (Search rollouts pilot the opponent's deck with this policy too, so
    unknown Pokemon must be allowed their real energy needs.)"""
    if card_id in ENERGY_QUOTA:
        return ENERGY_QUOTA[card_id]
    if card_id not in _quota_cache:
        data = _cards().get(card_id)
        _quota_cache[card_id] = max((_costs().get(a, 1) for a in data.attacks),
                                    default=1) if data is not None else 1
    return _quota_cache[card_id]


def _costs():
    global _attack_cost
    if _attack_cost is None:
        from ptcg.engine.sdk import api
        _attack_cost = {a.attackId: len(a.energies) for a in api.all_attack()}
    return _attack_cost


def _prize_worth(pokemon) -> int:
    data = _cards().get(_get(pokemon, "id"))
    if data is None:
        return 1
    return 3 if data.megaEx else 2 if data.ex else 1


_threat_cache: dict[str, int] = {}


def _future_prize_worth(card_id: int) -> int:
    """Best prize worth this card can evolve into (kill the egg before it
    hatches: a benched Riolu is a future 3-prize Mega Lucario)."""
    if not _threat_cache:
        by_name: dict[str, int] = {}
        for c in _cards().values():
            worth = 3 if c.megaEx else 2 if c.ex else 1
            by_name[c.name] = max(by_name.get(c.name, 1), worth)
        for _ in range(2):  # propagate through two-stage lines
            for c in _cards().values():
                base = c.evolvesFrom
                if base:
                    mine = _threat_cache.get(c.name, by_name.get(c.name, 1))
                    if mine > _threat_cache.get(base, 0):
                        _threat_cache[base] = mine
    data = _cards().get(card_id)
    if data is None:
        return 1
    return max(_threat_cache.get(data.name, 1),
               3 if data.megaEx else 2 if data.ex else 1)


def _target_score(pokemon) -> float:
    """How much we want to KO this opponent Pokemon (pixiux's prize economy):
    prizes first, then denial — invested energy and evolvable basics are their
    future offense — then evolution stage and HP."""
    data = _cards().get(_get(pokemon, "id"))
    energies = len(_get(pokemon, "energies") or [])
    score = _prize_worth(pokemon) * 1000
    score += energies * 250
    score += len(_get(pokemon, "tools") or []) * 100
    if data is not None:
        if data.stage2:
            score += 250
        elif data.stage1:
            score += 130
        elif energies >= 1:  # an invested basic is their next evolution line
            score += 350
        # Deny future multi-prize attackers while they're still small.
        score += (_future_prize_worth(_get(pokemon, "id")) - _prize_worth(pokemon)) * 250
    return score + (_get(pokemon, "hp") or 0)


def _damage(attack_id: int, hand_count: int, attacker_type, defender) -> int:
    if attack_id == POWERFUL_HAND:
        return 20 * hand_count  # damage counters: weakness does not apply
    dmg = ATTACK_DAMAGE.get(attack_id, 0)
    data = _cards().get(_get(defender, "id"))
    if data is not None:
        if data.weakness == attacker_type:
            dmg *= 2
        elif data.resistance == attacker_type:
            dmg = max(0, dmg - 30)
    return dmg


class Plan:
    """Best executable (attacker, attack, target) this turn. Board indexes:
    0 = active, i+1 = bench slot i. -1 = no plan."""

    def __init__(self):
        self.attacker = -1
        self.target = -1
        self.attack_id = None
        self.needs_energy = False
        self.lethal = False
        self.winning = False


class AlakazamPolicy:
    def __init__(self, obs_select, current):
        self.select = obs_select
        self.state = current
        self.context = _get(obs_select, "context")
        self.me_idx = _get(current, "yourIndex") if current is not None else 0
        players = _get(current, "players") if current is not None else None
        self.me = players[self.me_idx] if players else None
        self.opp = players[1 - self.me_idx] if players else None
        self._plan = None

        self.hand_ids = [_get(c, "id") for c in (_get(self.me, "hand") or [])] \
            if self.me is not None else []
        self.hand_count = len(self.hand_ids)
        self.can_gust = (BOSS in self.hand_ids
                         and not _get(self.state, "supporterPlayed"))
        self.can_switch = any(_get(o, "type") == OPT_RETREAT
                              for o in (_get(obs_select, "option") or []))

    @property
    def plan(self) -> Plan:
        # Lazy: most selects (yes/no, fetches, discards) never consult the
        # plan, and this is the policy's one expensive computation — building
        # it eagerly measurably slowed search rollouts.
        if self._plan is None:
            self._plan = Plan()
            if self.me is not None:
                self._plan_attack()
        return self._plan

    def _board(self, player) -> list:
        return list(_get(player, "active") or []) + list(_get(player, "bench") or [])

    def _field_count(self, card_id: int) -> int:
        return sum(1 for p in self._board(self.me)
                   if p is not None and _get(p, "id") == card_id)

    def _low_deck(self) -> bool:
        return (_get(self.me, "deckCount") or 60) <= LOW_DECK_COUNT

    def _plan_attack(self) -> None:
        if (_get(self.state, "turn") or 0) < 2:
            return
        hand_energy = any(c in ENERGY_IDS for c in self.hand_ids)
        energy_attached = bool(_get(self.state, "energyAttached"))
        best_score = -1.0
        for a_idx, mine in enumerate(self._board(self.me)):
            if mine is None:
                continue
            if a_idx != 0 and not self.can_switch:
                break
            data = _cards().get(_get(mine, "id"))
            if data is None:
                continue
            have = len(_get(mine, "energies") or [])
            for attack_id in data.attacks:
                need = _costs().get(attack_id, 99)
                needs_energy = False
                if have < need:
                    if hand_energy and not energy_attached and have + 1 >= need:
                        needs_energy = True
                    else:
                        continue
                for t_idx, target in enumerate(self._board(self.opp)):
                    if target is None:
                        continue
                    if t_idx != 0 and not self.can_gust:
                        break
                    # Cards we must spend from hand before attacking shrink
                    # Powerful Hand: the energy attach and Boss's Orders.
                    hand_after = (self.hand_count - (1 if needs_energy else 0)
                                  - (1 if t_idx != 0 else 0))
                    dmg = _damage(attack_id, hand_after, data.energyType, target)
                    if dmg <= 0:
                        continue
                    hp = _get(target, "hp") or 1
                    score = _target_score(target)
                    prize = _prize_worth(target) if dmg >= hp else 0
                    if prize == 0:
                        score *= dmg / hp
                    winning = prize >= len(_get(self.opp, "prize") or [1] * 6)
                    if winning:
                        score = 50000
                    score += 220 if a_idx == 0 else 0
                    score += 300 if t_idx == 0 else 0
                    score += have
                    if score > best_score:
                        best_score = score
                        self._plan.attacker = a_idx
                        self._plan.target = t_idx
                        self._plan.attack_id = attack_id
                        self._plan.needs_energy = needs_energy
                        self._plan.lethal = prize > 0
                        self._plan.winning = winning

    # ---- option scoring (single scalar, higher = chosen first) ----

    def choose(self) -> list[int]:
        options = _get(self.select, "option") or []
        max_count = _get(self.select, "maxCount") or 0
        if not options or max_count == 0:
            return []
        scores = [self._score_option(i, o) for i, o in enumerate(options)]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda t: -t[1])]
        return ranked[:max_count]

    def _score_option(self, position: int, option) -> float:
        kind = _get(option, "type")
        if kind == OPT_NUMBER:
            return _get(option, "number") or 0
        if kind == OPT_YES:
            if self.context == CTX_IS_FIRST:
                # Go second: first supporter and first attack beat first evolve
                # for a draw-engine deck. (A/B'd vs going first.)
                return -1
            return 1
        if kind == OPT_NO:
            return 0
        if kind == OPT_CARD:
            return self._score_card(option)
        if kind == OPT_PLAY:
            return self._score_play(option)
        if kind == OPT_ATTACH:
            return self._score_attach(option)
        if kind == OPT_EVOLVE:
            # Evolving draws cards here (Psychic Draw), so it is almost always
            # right; keep energy-holding targets first.
            target = self._in_play(option)
            return 9000 + len(_get(target, "energies") or []) if target is not None else 9000
        if kind == OPT_ABILITY:
            # Run Away Draw etc: free cards. Skip when it would deck us out,
            # and don't let Dudunsparce shuffle itself off a thin board.
            if self._low_deck():
                return -1
            card = self._card_at(option)
            if (card is not None and _get(card, "id") == DUDUNSPARCE
                    and sum(1 for p in self._board(self.me) if p is not None) <= 2):
                return -1
            return 30000
        if kind == OPT_RETREAT:
            return 2000 if self.plan.attacker >= 1 else -1
        if kind == OPT_ATTACK:
            return self._score_attack(option)
        if kind == OPT_END:
            return -0.5
        return 0

    def _score_attack(self, option) -> float:
        matches = _get(option, "attackId") == self.plan.attack_id
        if matches and self.plan.attacker == 0 and not self.plan.needs_energy:
            if self.plan.target == 0:
                if self.plan.winning:
                    return 100000
                if self.plan.lethal:
                    # Take the KO before anything else can shrink our hand.
                    return 25000
            # Bench target: let Boss's Orders (3200) fire first.
        return 1000 + (100 if matches else 0)

    def _score_play(self, option) -> float:
        card = self._hand_card(option)
        card_id = _get(card, "id") if card is not None else None
        data = _cards().get(card_id)
        if data is not None and (data.hp or data.attacks):  # a Pokemon: bench it
            return 20000
        if card_id == RARE_CANDY:
            return 9500
        if card_id == POFFIN:
            board = sum(1 for p in self._board(self.me) if p is not None)
            return 5500 if board < 4 else 1500
        if card_id == POKE_PAD:
            return -1 if self._low_deck() else 5000
        if card_id == DAWN:
            return -1 if self._low_deck() else 3050
        if card_id == HILDA:
            return -1 if self._low_deck() else 3000
        if card_id == BOSS:
            return 3200 if self.plan.target >= 1 else -1
        if card_id == ENH_HAMMER:
            return 2600
        if card_id in (NIGHT_STRETCHER, LANAS_AID, SACRED_ASH):
            return 2200
        if card_id == BATTLE_CAGE:
            return 1100
        return 2500  # unknown trainer: hand is damage, don't dump eagerly

    def _score_attach(self, option) -> float:
        pokemon = self._in_play(option)
        if pokemon is None:
            return 0
        board_index = (_get(option, "inPlayIndex") or 0) \
            + (0 if _get(option, "inPlayArea") == AREA_ACTIVE else 1)
        score = self._energy_target_score(pokemon,
                                          _get(option, "inPlayArea") == AREA_ACTIVE)
        if board_index == self.plan.attacker and self.plan.needs_energy:
            score += 200
        return score

    def _energy_target_score(self, pokemon, active: bool) -> float:
        card_id = _get(pokemon, "id")
        have = len(_get(pokemon, "energies") or [])
        if have >= _quota(card_id):
            # Everyone is fed: keep the energy in hand, it's 20 damage.
            return -1
        score = 8000 + (10 if active else 0)
        score += {ALAKAZAM: 120, KADABRA: 100, ABRA: 80, DUNSPARCE: 20}.get(card_id, 0)
        return score

    def _score_card(self, option) -> float:
        card = self._card_at(option)
        if self.context in (CTX_SWITCH, CTX_TO_ACTIVE):
            if _get(option, "playerIndex") != self.me_idx:
                # Boss's Orders target: drag the plan's victim.
                bonus = 100 if _get(option, "index") == self.plan.target - 1 else 0
                return bonus + (_target_score(card) * 0.01 if card is not None else 0)
            return self._score_promote(option, card)
        if self.context == CTX_SETUP_ACTIVE:
            # Dunsparce soaks up front; Abras stay back to evolve.
            return {DUNSPARCE: 4, ABRA: 1}.get(_get(card, "id"), 0) if card is not None else 0
        if self.context == CTX_SETUP_BENCH:
            return 10
        if self.context == CTX_TO_BENCH:
            # Poffin fetches: Abras become Alakazams, everything else is filler.
            return {ABRA: 100, DUNSPARCE: 50}.get(_get(card, "id"), 10) \
                if card is not None else 0
        if self.context == CTX_TO_HAND:
            return self._score_to_hand(card)
        if self.context in (CTX_TO_DECK, CTX_TO_DECK_BOTTOM):
            # Dual-use: from hand it's "return the junk", but from the discard
            # it's recovery (Night Stretcher / Sacred Ash) — send our best
            # cards back to the deck so we can draw them again.
            if _get(option, "area") == AREA_DISCARD:
                return self._score_to_hand(card)
            return self._score_discard(card)
        if self.context == CTX_DISCARD:
            return self._score_discard(card)
        if self.context == CTX_ATTACH_FROM:
            return self._energy_target_score(card, _get(option, "area") == AREA_ACTIVE) \
                if card is not None else 0
        return 0

    def _score_promote(self, option, card) -> float:
        """New active after retreat/KO: the plan attacker, else whoever hits
        hardest from the front next turn (counting one attach from hand —
        a 0-energy Alakazam plus an attach is a full Powerful Hand, while
        anything that can't attack is a wasted turn we can't afford)."""
        if card is None:
            return 0
        if _get(option, "index") == self.plan.attacker - 1:
            return 10000
        data = _cards().get(_get(card, "id"))
        if data is None:
            return 0
        have = len(_get(card, "energies") or [])
        hand_energy = any(c in ENERGY_IDS for c in self.hand_ids)
        opp_active = (self._board(self.opp) or [None])[0]
        best = 0.0
        for attack_id in data.attacks:
            missing = _costs().get(attack_id, 99) - have
            if missing > 1 or (missing == 1 and not hand_energy):
                continue
            dmg = _damage(attack_id, self.hand_count, data.energyType, opp_active)
            best = max(best, dmg + (40 if missing <= 0 else 0))
        return best + (_get(card, "hp") or 0) * 0.05

    def _score_to_hand(self, card) -> float:
        if card is None:
            return 0
        card_id = _get(card, "id")
        score = 200.0 - self.hand_ids.count(card_id) * 100
        if card_id == ALAKAZAM:
            score += 60
        elif card_id == KADABRA:
            score += 30
        elif card_id == RARE_CANDY:
            score += 50
        elif card_id == ABRA:
            score += 20 if self._field_count(ABRA) < 2 else -50
        elif card_id in ENERGY_IDS:
            score += 40 if not any(c in ENERGY_IDS for c in self.hand_ids) else 0
        elif card_id in (DAWN, HILDA, POKE_PAD):
            score += 10
        return score

    def _score_discard(self, card) -> float:
        """Forced discards: shed situational trainers, keep the line."""
        if card is None:
            return 50
        order = {SACRED_ASH: 40, LANAS_AID: 38, BATTLE_CAGE: 35, ENH_HAMMER: 30,
                 NIGHT_STRETCHER: 25, HILDA: 20, DAWN: 18, POKE_PAD: 15,
                 POFFIN: 12, BOSS: 8, DUNSPARCE: 5, DUDUNSPARCE: 5,
                 RARE_CANDY: 4, ABRA: 2, KADABRA: 2, ALAKAZAM: 0}
        card_id = _get(card, "id")
        if card_id in ENERGY_IDS:
            return 6
        return order.get(card_id, 22)

    # ---- card lookups ----

    def _hand_card(self, option):
        try:
            return (_get(self.me, "hand") or [])[_get(option, "index")]
        except (IndexError, TypeError):
            return None

    def _in_play(self, option):
        zone = _get(self.me, "active") if _get(option, "inPlayArea") == AREA_ACTIVE \
            else _get(self.me, "bench")
        try:
            return (zone or [])[_get(option, "inPlayIndex")]
        except (IndexError, TypeError):
            return None

    def _card_at(self, option):
        player_index = _get(option, "playerIndex")
        players = _get(self.state, "players") if self.state is not None else None
        if players is None or player_index is None:
            return None
        player = players[player_index]
        area = _get(option, "area")
        zone = {AREA_HAND: _get(player, "hand"), AREA_DISCARD: _get(player, "discard"),
                AREA_ACTIVE: _get(player, "active"), AREA_BENCH: _get(player, "bench"),
                AREA_PRIZE: _get(player, "prize")}.get(area)
        try:
            return (zone or [])[_get(option, "index")]
        except (IndexError, TypeError):
            return None


def choose(select, current) -> list[int]:
    return AlakazamPolicy(select, current).choose()


def make_agent(deck: list[int] | None = None):
    deck = deck or DECK

    def agent(obs: dict) -> list[int]:
        if obs["select"] is None:
            return deck
        return choose(obs["select"], obs.get("current"))

    return agent


agent = make_agent()
