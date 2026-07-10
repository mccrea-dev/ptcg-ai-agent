# cabt Engine notes

What we've verified about the competition engine, from the bundled source
(`kaggle_environments/envs/cabt/`) and the [official docs](https://matsuoinstitute.github.io/cabt/).
Everything below ran locally on 2026-07-09 (Windows, Python 3.11, kaggle-environments 1.30.1).

## Agent contract

- `agent(obs) -> list[int]`.
- **First call**: `obs["select"] is None` → return your 60-card deck as card IDs.
  Basic energy is exempt from the 4-copy limit (sample deck runs 33× card 3).
- **Every later call**: return between `select["minCount"]` and `select["maxCount"]`
  indices into `select["option"]`. The built-in random agent always returns exactly
  `maxCount`.
- Any exception, illegal selection, or malformed deck → status `INVALID`/`ERROR`,
  reward −1. The opponent immediately wins. Crash-proofing is mandatory.

## Observation

- `select`: `type`, `context`, `option` (list), `minCount`, `maxCount`, plus
  situational fields (`deck` when picking from deck, `contextCard`, `effect`,
  `remainDamageCounter`, `remainEnergyCost`).
- `current`: `turn`, `yourIndex`, per-turn flags (`supporterPlayed`,
  `energyAttached`, `retreated`, ...), `stadium`, and two `players` entries:
  active/bench Pokémon (hp, energies, tools, status), own `hand` (opponent's is
  `None` — only `handCount`), `discard`, `prize` (face-down slots are `None`),
  `deckCount`.
- `logs`: typed event records (DRAW, PLAY, ATTACK, HP_CHANGE, COIN, ...) — the
  opponent's visible actions live here.
- `search_begin_input`: serialized state blob to seed the native search API.

## Native search API (M2)

The C++ lib exports a search mode: `search_begin(agent_observation, your_deck,
your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active, ...)`
→ `SearchState{observation, searchId}`, then `search_step(search_id, select)` to
advance, `search_end()`/`search_release()` to free. You supply *guesses* for the
hidden zones — i.e., the engine natively supports determinized lookahead. The pip
package doesn't ship the Python wrappers for this (only `battle_*`); they're in
the official SDK on Kaggle, or we bind `lib.SearchBegin/Step/End` ourselves.

## Timing & episode mechanics

- No per-move timeout (`actTimeout: 0`); one shared pool per player:
  `remainingOverageTime: 600` seconds for the whole game. Overrun → TIMEOUT loss.
- Rewards: win +1 / loss −1 / draw 0. Step cap 10000.
- A random-vs-random game ran 88 select-steps; expect a few hundred per real game.

## Still unverified (needs Kaggle account / rules page)

- Episode-worker hardware (CPU/RAM/Python version) and bundle size limit.
- Tournament rule adjustments vs. official TCG rules.
- Card pool list and card metadata access (`all_card_data()` is in the official
  SDK download, not the pip package).
