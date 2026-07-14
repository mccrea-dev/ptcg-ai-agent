"""Flat Monte Carlo search agent.

At each main-phase decision on our own turn, candidate options are evaluated
by determinized rollouts: sample the hidden zones, play the option, then let
the tactical policy finish the game for both sides through the native search
API. The option with the best average outcome wins. Multi-pick selects and
forced moves are answered instantly by the tactical policy.

Clock handling: Kaggle gives each player one 600-second overage pool for the
whole game and running out is an instant loss. Each decision gets a slice of
what remains; below the reserve we stop searching entirely and play the
tactical policy, which answers in microseconds.
"""

import os
import random
import time

from ptcg.agents import priority_agent, tactical
from ptcg.agents.search import determinize
from ptcg.decks import load_deck
from ptcg.engine.sdk import api

DECK = load_deck("starter")

SELECT_MAIN = 0
MAX_ROLLOUT_STEPS = 3000


def _policy_on_search_obs(observation) -> list[int]:
    """Run the tactical policy on a search-API Observation dataclass (its
    accessors tolerate dataclasses directly)."""
    return tactical.choose(observation.select, observation.current)


def _truncated_value(cur, me: int) -> float:
    """Board-aware value for a rollout cut off mid-game, in (-1, 1) so real
    results (+/-1) always dominate. Prizes are the game, powered attackers
    and board width are tempo, hand is resources (and literal damage for
    hand-scaling decks). Profiling showed rollout cost is ~all engine steps,
    so a richer eval at a shorter horizon beats more steps."""
    mine, theirs = cur.players[me], cur.players[1 - me]

    def powered(player):
        return sum(1 for p in (player.active or []) + (player.bench or [])
                   if p is not None and len(p.energies or []) >= 1)

    def width(player):
        return sum(1 for p in (player.active or []) + (player.bench or [])
                   if p is not None)

    v = 0.60 * (len(theirs.prize) - len(mine.prize)) / 6.0
    v += 0.15 * max(-1.0, min(1.0, (powered(mine) - powered(theirs)) / 3.0))
    v += 0.10 * (width(mine) - width(theirs)) / 6.0
    v += 0.15 * max(-1.0, min(1.0, (len(mine.hand) - len(theirs.hand)) / 10.0))
    return v


def _rollout(root, first_action: list[int], policy=_policy_on_search_obs,
             max_steps: int = MAX_ROLLOUT_STEPS, eval_fn=None) -> float:
    """Play first_action from root, then policy-vs-policy to the end (or the
    step cap). Returns +1/-1 for a real result, else a board-aware value
    estimate — a truncated rollout with a sane eval buys several finished
    games' worth of signal per second."""
    me = root.observation.current.yourIndex
    state = api.search_step(root.searchId, first_action)
    steps = 0
    while state.observation.current.result < 0 and steps < max_steps:
        state = api.search_step(state.searchId, policy(state.observation))
        steps += 1
    cur = state.observation.current
    result = cur.result
    if result == me:
        return 1.0
    if result == 1 - me:
        return -1.0
    if result < 0:
        return (eval_fn or _truncated_value)(cur, me)
    return 0.0


def _decision_budget(remaining: float | None, default: float, reserve: float) -> float:
    """Seconds of thinking this decision may use."""
    if remaining is None:  # local runners don't track a clock
        return default
    if remaining <= reserve:
        return 0.0
    # Assume ~45 searched decisions still ahead; cap so no single decision
    # can eat the pool. Tunable, deliberately conservative.
    return min(8.0, (remaining - reserve) / 45)


def make_agent(deck: list[int] | None = None,
               opp_deck_guess: list[int] | None = None,
               seed: int | None = None,
               budget_seconds: float = 1.5,
               reserve_seconds: float = 60.0,
               max_options: int = 5,  # tactical ordering is strong; fewer
               # candidates = more rollouts each (was 10 with a weak prior)
               # Ladder replays showed a median of only 104s of the 600s pool
               # being spent: the old cap of 24 passes bound before the time
               # budget did. Passes are now effectively time-limited.
               max_passes: int = 150,
               main_phase_only: bool = False,
               halving_schedule: tuple[int, ...] = (),
               confidence_prune_z: float = 0.0,
               setup_search: bool = False,  # default off until its A/B reports
               ranker_path: str | None = None,  # trained BC net for root ordering
               net_rollouts: bool = False,  # net also pilots both sides in rollouts
               prior_anchor_z: float = 1.0,  # 0 = raw argmax over rollout means
               rollout_steps: int = 160,  # truncate + eval; ~4-6 turns of
               # lookahead at a fraction of a finished game's cost
               value_path: str | None = None):  # trained value net grades
               # truncated rollouts instead of the hand-tuned board eval
    # halving_schedule stays available for experiments, but defaults off:
    # pruning on 4 noisy rollouts measured 46.7% CI[34.6,59.1] vs uniform.
    # confidence_prune_z > 0 enables evidence-based pruning instead: an option
    # is dropped only once its mean + z*stderr falls below the leader's
    # mean - z*stderr, i.e. when we are statistically sure it lost the race.
    deck = deck or DECK
    # With no explicit guess, infer the opponent's deck from their revealed
    # cards each decision (mirror of our own deck until they show something).
    infer_opponent = opp_deck_guess is None
    rng = random.Random(seed)
    # PTCG_BUDGET_S pins the per-decision budget outright (used by the bundle
    # verifier); without it the real clock governs when one is present.
    env_budget = os.environ.get("PTCG_BUDGET_S")
    default_budget = float(env_budget) if env_budget else budget_seconds
    use_clock = env_budget is None
    # Card we chose at setup-active (context 1); the observation shows our own
    # active as null until turn 1, so this memory keeps the unseen-pool math
    # balanced and the setup-bench decisions searchable.
    remembered_active: list[int | None] = [None]
    fast_policy = tactical.make_agent(deck)
    ranker = None
    if ranker_path is not None:
        from ptcg.agents.learned import NumpyRanker
        ranker = NumpyRanker(ranker_path)
    eval_fn = None
    if value_path is not None:
        from ptcg.agents.learned import NumpyValue
        net = NumpyValue(value_path)

        def eval_fn(cur, me):
            # Net scores from the to-move player's view; flip to root's view.
            v = net.value(cur)
            return v if cur.yourIndex == me else -v

    rollout_policy = _policy_on_search_obs
    if net_rollouts and ranker is not None:
        import numpy as _np

        def rollout_policy(observation):
            sel = observation.select
            if sel.maxCount != 1 or len(sel.option) < 2:
                return _policy_on_search_obs(observation)
            try:
                return [int(_np.argmax(ranker.scores(
                    sel, observation.current, priority_agent.ATTACK_DAMAGE)))]
            except Exception:
                return _policy_on_search_obs(observation)

    def choose_and_remember(obs: dict, action: list[int]) -> list[int]:
        """Record our setup-active pick (context 1) on the way out, so the
        face-down window stays searchable."""
        sel = obs["select"]
        if sel.get("context") == 1 and action:
            opt = sel["option"][action[0]]
            hand = obs["current"]["players"][obs["current"]["yourIndex"]]["hand"]
            if opt.get("area") == 2 and opt.get("index") is not None and hand:
                remembered_active[0] = hand[opt["index"]]["id"]
        return action

    def agent(obs: dict) -> list[int]:
        if obs["select"] is None:
            return deck
        select = obs["select"]
        n = len(select["option"])
        # Search any single-pick decision (main phase, card targets, yes/no,
        # setup...); multi-pick selects and forced moves are answered fast.
        searchable = select["maxCount"] == 1 and n > 1
        if main_phase_only:
            searchable = searchable and select["type"] == SELECT_MAIN
        # During the setup-bench window our own active sits face-down and is
        # null even to us; without the memory of what we chose, determinization
        # would smuggle that card back into the deck — skip searching then.
        me = obs["current"]["yourIndex"]
        own_active = obs["current"]["players"][me]["active"]
        facedown = bool(own_active) and own_active[0] is None
        if facedown and (not setup_search or remembered_active[0] is None):
            searchable = False
        if not searchable:
            return choose_and_remember(obs, fast_policy(obs))

        remaining = obs.get("remainingOverageTime") if use_clock else None
        budget = _decision_budget(remaining, default_budget, reserve_seconds)
        if budget <= 0.0:
            return choose_and_remember(obs, fast_policy(obs))

        # Concentrate rollouts on the most plausible moves — ranked by the
        # trained expert-imitation net when available, else the tactical policy.
        options = select["option"]
        candidates = None
        if ranker is not None:
            try:
                net_scores = ranker.scores(select, obs["current"],
                                           priority_agent.ATTACK_DAMAGE)
                candidates = sorted(range(n), key=lambda i: -net_scores[i])
            except Exception:
                candidates = None
        if candidates is None:
            pol = tactical.AlakazamPolicy(select, obs["current"])
            candidates = sorted(range(n),
                                key=lambda i: -pol._score_option(i, options[i]))
        candidates = candidates[:max_options]

        if infer_opponent:
            from ptcg.agents.search import infer
            opp_guess = infer.guess_deck(obs, fallback=deck)
        else:
            opp_guess = opp_deck_guess

        obs_cls = api.to_observation_class(obs)
        # Endgame: with <=2 prizes on either side, games resolve within the
        # horizon anyway — take exact results instead of eval estimates. Cap
        # at 4x the normal horizon (not unbounded: a stall matchup's 3000-step
        # rollouts inside one pass could blow the clock pool right when it's
        # lowest).
        players = obs["current"]["players"]
        endgame = min(len(players[0]["prize"]), len(players[1]["prize"])) <= 2
        steps_cap = min(MAX_ROLLOUT_STEPS, rollout_steps * 4) if endgame \
            else rollout_steps
        survivors = list(candidates)
        sums = dict.fromkeys(candidates, 0.0)
        counts = dict.fromkeys(candidates, 0)
        passes = 0
        deadline = time.monotonic() + budget
        facedown_card = remembered_active[0] if facedown else None
        try:
            while passes < max_passes and time.monotonic() < deadline:
                det = determinize.sample(obs, deck, opp_guess, rng,
                                         my_facedown_active=facedown_card)
                # One root per determinization: search states persist, so every
                # survivor can branch off the same root before memory is reclaimed.
                root = api.search_begin(
                    obs_cls, det.your_deck, det.your_prize, det.opponent_deck,
                    det.opponent_prize, det.opponent_hand, det.opponent_active,
                )
                for i in survivors:
                    sums[i] += _rollout(root, [i], rollout_policy, steps_cap,
                                        eval_fn)
                    counts[i] += 1
                    # A pass is re-entrant on the clock: without this check a
                    # slow pass can overshoot the deadline by its full length.
                    if time.monotonic() >= deadline and min(counts.values()) > 0:
                        break
                api.search_end()
                passes += 1
                # Anchored focus: the decision rule only asks "does some
                # challenger beat the tactical move with confidence?" — after
                # a uniform look, spend the rest of the budget answering that
                # for the two best challengers instead of sampling everyone.
                if (prior_anchor_z > 0 and passes == 4 and len(survivors) > 3
                        and candidates[0] in survivors):
                    anchor = candidates[0]
                    challengers = sorted(
                        (i for i in survivors if i != anchor),
                        key=lambda i: sums[i] / counts[i], reverse=True)[:2]
                    survivors = [anchor] + challengers
                # Successive halving: drop the losing half and reinvest their
                # rollouts in the contenders.
                if passes in halving_schedule and len(survivors) > 2:
                    survivors.sort(key=lambda i: sums[i] / counts[i], reverse=True)
                    survivors = survivors[:max(2, len(survivors) // 2)]
                # Evidence-based pruning: cut only options whose confidence
                # interval has separated below the leader's.
                if confidence_prune_z > 0 and passes >= 4 and len(survivors) > 1:
                    def bounds(i):
                        n_i = counts[i]
                        mean = sums[i] / n_i
                        stderr = (max(1.0 - mean * mean, 0.0) / n_i) ** 0.5
                        return mean - confidence_prune_z * stderr, \
                               mean + confidence_prune_z * stderr
                    leader_low = max(bounds(i)[0] for i in survivors)
                    survivors = [i for i in survivors
                                 if bounds(i)[1] >= leader_low]
        except Exception:
            # Search must never cost us the game; fall back to the fast policy.
            try:
                api.search_end()
            except Exception:
                pass
            return choose_and_remember(obs, fast_policy(obs))
        if passes == 0:
            return choose_and_remember(obs, fast_policy(obs))
        if prior_anchor_z > 0 and candidates[0] in survivors:
            # Trust the tactical policy's choice unless a challenger's rollout
            # mean beats it with statistical separation. With few rollouts the
            # threshold is unreachable and we play the policy move; with many,
            # genuine improvements clear it. Search becomes strictly additive
            # (measured: raw argmax over 2-3 rollouts/option scored 39% vs the
            # policy's own 50% against pixiux — noise overriding a good prior).
            anchor = candidates[0]

            def mean_se(i):
                # Variance floored at 0.25: a short streak of ±1 outcomes must
                # not masquerade as certainty (at m=±1 the sample variance is 0).
                m = sums[i] / counts[i]
                return m, (max(1.0 - m * m, 0.25) / counts[i]) ** 0.5

            a_mean, a_se = mean_se(anchor)
            best, best_margin = anchor, 0.0
            for i in survivors:
                if i == anchor or counts[i] < 4:
                    continue
                m, se = mean_se(i)
                margin = (m - a_mean) - prior_anchor_z * (se + a_se)
                if margin > 0 and margin > best_margin:
                    best, best_margin = i, margin
            return choose_and_remember(obs, [best])
        return choose_and_remember(
            obs, [max(survivors, key=lambda i: sums[i] / counts[i])])

    return agent


agent = make_agent()
