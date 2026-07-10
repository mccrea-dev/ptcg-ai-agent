# ptcg

Agent for the [Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
(Kaggle, 2026). Plan and milestones: [ROADMAP.md](ROADMAP.md).

## Setup (fresh machine)

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 and all dependencies
(including the cabt game engine, bundled in `kaggle-environments`) are installed with:

```sh
uv sync
```

Kaggle auth (needed for data download, scouting, and submissions): create an
API token at kaggle.com → Settings → API, and save it to
`~/.kaggle/access_token`.

Then fetch the official competition data (card database, engine source, SDK —
gitignored, ~10MB):

```sh
uv run python tools/fetch_official.py
```

Optional restores, all regenerable: mined episodes via `ptcg.data.scout`,
the 86k-decision training set via `kaggle datasets download gabemccrea/ptcg-train-data`
or by re-running `ptcg.data.trajectories` over downloaded replays. Training
deps: `uv sync --group train` (torch).

## Verify

```sh
uv run pytest
```

Runs full games through the engine, including a zone-accounting audit of
every decision.

## Layout

- `ptcg/agents/` — one module per agent; every agent is `agent(obs) -> list[int]`
- `ptcg/decks/` — 60-card deck lists as CSV (one card ID per line)
- `ptcg/engine/` — local game runner around `kaggle_environments.make("cabt")`
- `tests/` — pytest suite
