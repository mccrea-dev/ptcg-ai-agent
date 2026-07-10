"""Run local cabt games between two agents."""

from dataclasses import dataclass

from kaggle_environments import make


@dataclass
class GameResult:
    reward0: float
    reward1: float
    status0: str
    status1: str
    steps: int


def play_game(agent0, agent1) -> GameResult:
    env = make("cabt")
    env.run([agent0, agent1])
    last = env.steps[-1]
    return GameResult(
        reward0=last[0].reward,
        reward1=last[1].reward,
        status0=last[0].status,
        status1=last[1].status,
        steps=len(env.steps),
    )


def play_games(agent0, agent1, n: int) -> list[GameResult]:
    return [play_game(agent0, agent1) for _ in range(n)]
