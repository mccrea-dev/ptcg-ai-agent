"""Direct-engine game runner.

Drives battles through the SDK's battle_start/battle_select instead of
kaggle_environments.make(), which skips per-step JSON plumbing and framework
overhead. Used for local evaluation and, later, search rollouts. The engine
holds one global battle pointer, so only one game may run per process at a time.
"""

from ptcg.engine.sdk import game


def play_game(policy0, policy1, deck0: list[int], deck1: list[int],
              on_step=None) -> int:
    """Play one game; returns 0 or 1 for the winner, -1 for a draw.

    Policies are agent(obs)->list[int] functions; they are never asked for a
    deck (obs["select"] is always set), so submission agents work unchanged.
    on_step(obs, action), if given, observes every decision as it is made.
    """
    obs, start = game.battle_start(deck0, deck1)
    if obs is None:
        raise ValueError(f"battle_start failed: player {start.errorPlayer} deck error")
    try:
        policies = (policy0, policy1)
        while obs["current"]["result"] < 0:
            mover = obs["current"]["yourIndex"]
            action = policies[mover](obs)
            if on_step is not None:
                on_step(obs, action)
            obs = game.battle_select(action)
        result = obs["current"]["result"]
        return result if result in (0, 1) else -1
    finally:
        game.battle_finish()
