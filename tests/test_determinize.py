from ptcg.agents import priority_agent
from ptcg.agents.search.determinize import _unseen_pool
from ptcg.engine.sdk import game


def test_unseen_pool_balances_through_full_game():
    """At every decision point, the cards we can't account for must exactly
    fill the hidden zones (deck + face-down prizes). Any drift means a zone
    is missing from the accounting and every determinization would be wrong."""
    deck = priority_agent.DECK
    obs, _ = game.battle_start(deck, deck)
    checked = 0
    try:
        while obs["current"]["result"] < 0:
            cur = obs["current"]
            me = cur["yourIndex"]
            pool = _unseen_pool(deck, cur, obs["select"], me, include_hand=True)
            hidden = cur["players"][me]["deckCount"] + sum(
                1 for c in cur["players"][me]["prize"] if c is None
            )
            active = cur["players"][me]["active"]
            if active and active[0] is None:
                # Setup-bench window: our own chosen active is face-down and
                # invisible even to us, so exactly one card can't be accounted.
                # mc_agent skips searching here for the same reason.
                assert len(pool) == hidden + 1
            else:
                assert len(pool) == hidden, (
                    f"turn {cur['turn']}, select type {obs['select']['type']}: "
                    f"pool {len(pool)} != hidden {hidden}"
                )
            checked += 1
            obs = game.battle_select(priority_agent.agent(obs))
    finally:
        game.battle_finish()
    assert checked > 30
