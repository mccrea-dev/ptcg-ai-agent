# ptcg

Agent for the [Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
(Kaggle, 2026). Plan and milestones: [ROADMAP.md](ROADMAP.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 and all dependencies
(including the cabt game engine, bundled in `kaggle-environments`) are installed with:

```sh
uv sync
```

## Verify

```sh
uv run pytest
```

Runs a full random-vs-random game through the engine.

## Layout

- `ptcg/agents/` — one module per agent; every agent is `agent(obs) -> list[int]`
- `ptcg/decks/` — 60-card deck lists as CSV (one card ID per line)
- `ptcg/engine/` — local game runner around `kaggle_environments.make("cabt")`
- `tests/` — pytest suite
