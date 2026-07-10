from ptcg.agents import priority_agent, random_agent
from ptcg.engine.fast import play_game as fast_play
from ptcg.engine.runner import play_game


def test_random_vs_random_completes():
    result = play_game(random_agent.agent, random_agent.agent)
    assert result.status0 == "DONE"
    assert result.status1 == "DONE"
    assert result.reward0 in (-1, 0, 1)
    assert result.reward0 == -result.reward1 or result.reward0 == 0


def test_fast_runner_completes():
    winner = fast_play(priority_agent.agent, random_agent.agent,
                       priority_agent.DECK, random_agent.DECK)
    assert winner in (-1, 0, 1)
