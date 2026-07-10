# Pokémon TCG AI Battle Challenge — Project Roadmap

Planning document, written 2026-07-09. No code exists yet; this is the agreed starting point.
Owner: Gabe. Mentor/lead: Claude. Status: awaiting answers to §10 before implementation.

---

## 1. Competition overview

Two linked Kaggle competitions run by The Pokémon Company with the Matsuo Institute
(University of Tokyo), announced June 16, 2026. Total prizes >$290k.

**Simulation Category** ([kaggle.com/competitions/pokemon-tcg-ai-battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle))
- Head-to-head ladder. Agents play automated matches 24/7; skill tracked by a
  Gaussian (TrueSkill-style) rating that pairs similarly rated agents.
- Up to **5 submissions/day**; only your **two most recent** submissions count
  for final standings.
- Entry/team-merger deadline **Aug 9, 2026**; final submissions **Aug 16, 2026**.
- Entering Simulation is **required** to compete in Strategy.

**Strategy Category** ([kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy))
- This is where the money is. A **≤2,000-word Kaggle Writeup** (plus media
  gallery) explaining the agent's design. Scoring: **70% model approach,
  20% deck concept, 10% report quality**, informed by simulation-phase
  performance and agent stability.
- Entry deadline **Sept 6, 2026**; final submission **Sept 13, 2026** (verify exact UTC).
- Top 8 teams: **$30,000 each** + advance to a live Final Stage in Japan
  (+$50k first, +$30k second).

**The game environment (cabt Engine)**
- Proprietary C++ simulator (`libcg.so` / `cg.dll`) with Python bindings,
  integrated with `kaggle-environments==1.30.1` (`make("cabt", ...)`).
  Docs: [matsuoinstitute.github.io/cabt](https://matsuoinstitute.github.io/cabt/).
- Rules are official Pokémon TCG **with tournament-specific adjustments** (must read).
- Card pool: **~2,000 Standard-format cards**. Teams build their own **60-card
  deck** (list of card IDs); the agent supplies it during a deck-selection phase.
- Agent contract: `agent(obs_dict) -> list[int]`. Observation = `logs` (event
  history), `current` (board state), `select` (**enumerated legal options**).
  You return indices into the options. **The engine does move generation for us.**
- **10 minutes total clock per player per game.** Timeout = immediate loss.
  A crash = loss. Stability and time management are first-class requirements.
- The sim module exposes native **SearchBegin/SearchStep/SearchEnd** bindings —
  the organizers ship a C++ search engine most competitors may ignore.
- An official **episodes index dataset** exists: every ladder game is downloadable.

**⚠ Timeline reality check:** the competition is already running. From today we have
**~5.5 weeks to the Simulation deadline** and **~9.5 weeks to the Strategy report**.
The plan below compresses learning into the build; there is no separate "study phase."

## 2. Technical challenge breakdown

1. **Imperfect information** — opponent's hand, both deck orders, and face-down
   prizes are hidden. Exact game-tree search is impossible; we approximate.
2. **Stochasticity** — shuffles, coin flips, prize draws. Single games are noisy;
   evaluation needs hundreds of games to distinguish agents.
3. **Long horizons, many micro-decisions** — a game is hundreds of `select` calls
   (attach, trainer, evolve, retreat, attack, discard choices…). Credit assignment
   is hard for RL; the 10-minute clock must be budgeted across all of them.
4. **Deck building is part of the AI** — agent quality and deck choice interact.
   Evidence from a public competitor repo ([ptcg-abc](https://github.com/wmh/ptcg-abc)):
   deck choice dominates, and simple consistent decks outperform combo decks when
   piloted by simple policies.
5. **A live, adapting metagame** — the ladder population shifts weekly; final
   standings depend on the meta during the last week, not today's.
6. **Engineering constraints** — never crash, never time out, package everything
   into an offline Kaggle bundle (no internet at episode time), unknown-but-limited
   CPU/RAM on episode workers (verify; assume no GPU at inference).

## 3. AI concepts to learn (and why), in learning order

Each is introduced at the milestone that needs it — intuition first, then the
technical version, then how it applies to Pokémon TCG, then what we build with it.

1. **Agent loop & Markov decision processes** (M0) — the observe→act→reward frame
   everything else builds on. In TCG terms: obs = board+log, action = option index,
   reward = win/loss.
2. **State evaluation & greedy policies** (M1) — scoring a board (prizes taken,
   energy tempo, damage on board, cards in hand) and picking the action that
   maximizes it. This is the floor every later system must beat.
3. **Game-tree search: expectimax → MCTS** (M2) — looking ahead instead of
   evaluating only the present. Monte Carlo Tree Search handles huge branching
   and chance nodes by sampling; it's the backbone of AlphaGo/AlphaZero.
4. **Imperfect-information workarounds: determinization / PIMC / IS-MCTS** (M2) —
   sample plausible hidden states (opponent hand, deck order), solve each sampled
   "perfect-info" game, vote. Known weaknesses (strategy fusion) matter less in
   TCGs than in bluffing games like poker.
5. **Supervised/imitation learning on trajectories** (M3) — learn a policy from
   (state, action) pairs mined from top ladder agents' episodes. The proven Kaggle
   simulation-competition play (used by winners in Connect X, Lux AI, etc.).
6. **Neural state encoding** (M3) — turning a variable-size board with ~2,000
   card types into tensors: card embeddings + pointer/attention over the option
   list (the action space is "choose one of these N options," which varies per turn).
7. **Value networks + search hybrids (AlphaZero-lite)** (M3/M4) — replace slow
   rollouts with a learned evaluator inside MCTS. Optionally **oracle distillation**
   (train with hidden info visible, distill to the partial-obs agent — the Suphx
   mahjong trick).
8. **Self-play RL (PPO with action masking)** (M4, stretch) — only after IL works.
   RL from scratch in 5 weeks as a first project is the classic failure mode.
9. **Evaluation statistics & ratings** (M1 onward) — win-rate confidence intervals,
   TrueSkill/Elo, why 50 games tells you almost nothing.
10. **Ablation methodology & technical writing** (M5) — the Strategy report is 70%
    "model approach": judges reward principled design choices proven by ablations.

## 4. Approach analysis — what wins, what's a dead end

| Approach | Verdict | Why |
|---|---|---|
| Rule-based heuristic + consistent deck | **Build first** | Proven mid-ladder (Elo ~800 in public repo). Fast, stable, our baseline and fallback. |
| Determinized MCTS / engine's native search | **Core bet** | Engine ships C++ search bindings + fast sim; 10-min clock rewards fast lookahead. Likely under-used by competitors. |
| Imitation learning from ladder episodes | **Core bet** | Official episodes dataset + documented winning pattern in Kaggle sim comps. Bootstraps a NN policy without RL pain. |
| Value-net-guided search (AlphaZero-lite) | **Stretch goal** | Best "model approach" story for the 70% criterion; only after IL/search work. |
| Pure self-play deep RL from scratch | Avoid as primary | Compute- and expertise-hungry; poor fit for a 5-week window and a first agent project. |
| CFR / poker-style equilibrium solvers | Dead end | Game far too large; hidden-info structure doesn't demand it. |
| LLM agent at inference time | Dead end | No internet in episodes, latency vs. 10-min clock. (LLMs fine as dev tools.) |
| Complex combo decks | Dead end early | Public evidence: weak pilots waste strong decks. Revisit only if our policy gets strong. |

**Competitive edges to exploit** (non-generic, in priority order):
1. **Native search bindings** — if `SearchBegin/Step/End` is usable, we get C++-speed
   lookahead while others write Python rollouts.
2. **Episode mining + opponent/meta modeling** — scrape the ladder weekly; know the
   top decks; pick/tune our deck against the *final-week* meta, not June's.
3. **Deck–agent co-design** — evaluate decks *with our actual policy*, not in the
   abstract; ablate deck changes via automated round-robins.
4. **Time-budget-aware search** — spend seconds on pivotal turns (attack/KO math),
   milliseconds on forced choices; most agents will spend uniformly or too safely.
5. **Deck portfolio / mixed strategy** — the agent chooses its deck at game start;
   a randomized 2–3 deck portfolio is harder to counter-meta (verify rules allow it).
6. **Oracle distillation** — cheap to try once IL infra exists; a strong originality
   point for the report.

## 5. Milestones

**M0 — Entry, environment, first submission (Jul 9–16) · difficulty: easy · deps: none**
- Objectives: Kaggle entry accepted (both categories), rules read line-by-line
  (record answers to §9 unknowns), SDK + card data downloaded, random-vs-random
  game running locally, and a trivial agent submitted to the ladder.
- Concepts: agent loop / MDP framing; the cabt observation & option structures;
  Pokémon TCG rules (play ~10 human games on PTCG Live to build intuition).
- Deliverables: working dev environment; `random_agent` on the leaderboard; a
  RULES_NOTES.md of constraints; repo skeleton (§6).
- Success: our episode appears on the public ladder without crash/timeout.

**M1 — Heuristic agent + eval harness (Jul 14–23) · difficulty: moderate · deps: M0**
- Objectives: a hand-written policy playing a simple consistent deck (single main
  attacker, heavy draw/search trainers); local evaluation harness that runs
  N-game matches and computes ratings with confidence intervals; replay logging.
- Concepts: board evaluation functions; greedy vs. lookahead; evaluation statistics.
- Deliverables: `heuristic_agent` ≥95% vs. random, submitted; eval harness; first
  deck ablation (2–3 candidate decks compared with the same policy).
- Success: measurably above the ladder median; harness reproduces ladder ordering
  of our own agents.

**M2 — Search agent (Jul 20 – Aug 3) · difficulty: hard · deps: M1**
- Objectives: lookahead via the engine (native search bindings if usable, else
  `battle_start/battle_select` rollouts); determinization for hidden info; strict
  time manager (never exceed budget, degrade to heuristic under pressure).
- Concepts: expectimax, MCTS, PIMC/IS-MCTS, time control.
- Deliverables: `search_agent` beating `heuristic_agent` ≥60% over ≥400 games;
  profiling notes on sim speed (games/sec) — this number gates everything later.
- Success: clear ladder rating gain over M1 without any timeout losses.

**M3 — Learning from episodes (Jul 27 – Aug 10) · difficulty: hard · deps: M1 (harness), M2 useful**
- Objectives: episode scraper for the official index dataset (filter: high-rated
  agents, recent engine version, winner's moves); state/option encoder; behavior-cloned
  policy net; integrate as (a) standalone policy and (b) prior/evaluator inside search.
- Concepts: imitation learning, card embeddings, attention over variable option
  lists, action masking; optionally value nets and oracle distillation.
- Deliverables: dataset pipeline; `il_agent`; hybrid `search+net` agent; meta report
  (what decks/lines the top ladder agents play).
- Success: hybrid beats pure search head-to-head and on ladder.

**M4 — Ladder endgame (Aug 10–16) · difficulty: moderate · deps: M2/M3**
- Objectives: freeze the best agent; crash-proof wrapper (catch-all → heuristic
  fallback); final deck choice vs. observed meta; submission-slot strategy for the
  final days (two most recent count — don't experiment late).
- Deliverables: final two Simulation submissions; frozen code tag.
- Success: peak rating stable through the final window.

**M5 — Strategy report (Aug 17 – Sep 13) · difficulty: moderate · deps: all**
- Objectives: ablations that justify each component (search on/off, net on/off,
  deck variants, time-manager variants); visualizations (rating curves, win-rate
  matrices, cabt-viewer board states); the 2,000-word writeup targeting the
  70/20/10 rubric; media gallery.
- Deliverables: ablation table with confidence intervals; final Writeup submitted
  by Sept 6 entry / Sept 13 final deadlines.
- Success: report a domain-expert reviewer finds novel, rigorous, and reproducible.

## 6. Software architecture

Monorepo, single Python package. Structure earns its place only when a milestone
needs it — start with the skeleton, grow rightward.

```
poke/
├── ROADMAP.md                  # this file
├── pyproject.toml              # uv-managed; pinned deps incl. kaggle-environments==1.30.1
├── configs/                    # plain YAML per experiment; no framework until it hurts
├── ptcg/                       # the package
│   ├── engine/                 # ONLY module that touches cabt/kaggle-environments
│   │   ├── wrapper.py          # typed Observation/Option/GameResult dataclasses
│   │   └── runner.py           # play N games headless, seeds, timing hooks
│   ├── agents/                 # one module per agent, all implementing Agent protocol
│   │   ├── base.py             # Agent: choose(obs) -> list[int]; select_deck() -> list[str]
│   │   ├── random_agent.py
│   │   ├── heuristic/          # M1: evaluator + policy rules
│   │   ├── search/             # M2: determinizer, MCTS, time_manager
│   │   └── neural/             # M3: encoder, policy/value nets, hybrid glue
│   ├── decks/                  # deck lists (CSV of card IDs) + deck-building notes
│   ├── data/                   # M3: episode scraper, trajectory extraction, datasets
│   ├── training/               # M3: BC training loop; later self-play
│   ├── eval/                   # match runner, ratings (TrueSkill), significance tests
│   └── submission/             # bundle builder: package agent → tar.gz + smoke test
├── replays/                    # local game logs, JSONL, git-ignored
├── experiments/                # one folder per run: config copy + metrics + notes
├── tests/                      # pytest; engine wrapper contract tests, agent smoke tests
└── tools/                      # cabt-viewer integration, log analysis scripts
```

Decisions and reasons:
- **Engine isolation** (`ptcg/engine/`): the cabt API is proprietary and may change
  between engine updates; everything else codes against our dataclasses so an engine
  update touches one module.
- **Agent protocol**: every milestone produces an `Agent` with the same interface,
  so the eval harness, submission packager, and ladder pipeline never change.
- **Submission packager with smoke test**: a bundle that imports cleanly inside the
  `kaggle-environments` runtime, offline, under time limits — tested *before* burning
  one of 5 daily slots. Crashes and timeouts are losses; this is our safety net.
- **Replay storage as JSONL** (one game per line, obs/action/timing per step): flat
  files, greppable, feeds both debugging and M3 training. No database.
- **Experiment tracking**: start with per-run folders + a CSV of results; add
  Weights & Biases only when NN training starts (M3), where live loss curves pay off.
- **Config**: YAML + dataclasses. Hydra/omegaconf deferred — complexity not yet earned.
- **Testing strategy**: contract tests on the engine wrapper (options always valid,
  state parses on every phase), property test "agent never returns an illegal index,"
  golden-replay regression tests for the heuristic, and a full-game smoke test in CI.
- **Logging**: stdlib `logging`, structured per-move records into the replay files;
  agent decisions log their evaluation scores so losses are diagnosable.

## 7. Technology stack

- **Python 3.11** (match Kaggle runtime; verify exact version), **uv** for env/deps.
- **kaggle-environments==1.30.1** + official cabt SDK/card data from Kaggle.
- **numpy** now; **PyTorch** from M3 (training on GPU, inference must run CPU-only).
- **pytest**, **ruff** (lint+format), **GitHub private repo** + Actions CI (private
  until the deadline: our code is competitive advantage).
- **trueskill** package for local ratings; **pandas** for episode analysis.
- **Weights & Biases** free tier from M3.
- **Platform: WSL2 Ubuntu** for primary dev (`libcg.so`, RL tooling, and Kaggle
  parity are Linux-native; `cg.dll` exists but we don't want Windows-only bugs).
  Confirm in §10.

## 8. First-month prioritized task list (Jul 9 – Aug 9)

Week 1: Kaggle entry ✚ rules deep-read → dev env (WSL2, uv, SDK download) →
random game locally → repo skeleton + engine wrapper → **submit random agent**.
Week 2: obs/option exploration notebook → heuristic evaluator v1 → first real deck
→ eval harness with ratings → **submit heuristic agent** → deck A/B on ladder.
Week 3: sim speed profiling → native search bindings spike (go/no-go) →
determinized search v1 → time manager → **submit search agent**.
Week 4: episode scraper + meta analysis → encoder + BC training v1 → hybrid agent
→ **submit hybrid** → decide final-week focus (deck meta vs. model gains).
Throughout: every submission logged in experiments/; never leave a day's 5 slots
entirely unused once agents are stable — ladder time is data.

## 9. Risks and unknowns

| # | Risk / unknown | Impact | Mitigation |
|---|---|---|---|
| 1 | **Only ~5.5 weeks to Sim deadline** | High | Compressed plan above; heuristic agent is the always-shippable floor. |
| 2 | Episode-worker hardware/limits unknown (CPU, RAM, bundle size, Python ver) | High | Read rules page + forum in M0; design inference CPU-only and small. |
| 3 | Sim speed unknown → search/RL feasibility | High | Profile in week 3 before committing; native bindings spike is the go/no-go. |
| 4 | Native search bindings may be undocumented/unusable | Medium | Fallback: Python-level battle_start/select rollouts; budget the spike at 2 days. |
| 5 | Engine updates mid-competition invalidating models/data | Medium | Engine-isolation layer; re-scrape episodes after updates; keep heuristic current. |
| 6 | Tournament rule adjustments differ from the TCG we study | Medium | Read the adjusted rules first; trust the engine's legal-move list over intuition. |
| 7 | First NN project under deadline (footguns: encoding bugs, silent training failures) | Medium | IL before RL; overfit-one-batch sanity checks; hybrid design keeps search floor. |
| 8 | Final-week meta shift tanks our deck | Medium | Weekly meta scrapes; deck decision deferred to M4; portfolio idea if legal. |
| 9 | Rating variance / two-most-recent rule misplay | Low | No experimental submissions in the final 72h. |
| 10 | Team-of-one bandwidth | ? | Depends on §10 answers; milestones are cuttable at M3 boundary. |

## 10. Questions to answer before implementation

1. **Time budget**: how many hours/week can you commit through Aug 16, and is that
   sustainable? The plan assumes roughly 15–20.
2. **Kaggle account & entry**: do you have an account, and can you accept both
   competitions' rules this week? (Sim entry gates everything, deadline Aug 9.)
3. **Solo or team?** Up to 5 people; merger deadlines apply. Recruiting one
   experienced ML person would de-risk M3 considerably.
4. **Hardware**: what GPU (if any) do you have locally, and is WSL2 acceptable as
   the primary dev environment? Any budget for cloud GPU hours in M3?
5. **Pokémon TCG familiarity**: have you played the modern game? If not, plan
   ~5 hours on PTCG Live in week 1 — deck intuition is 20% of the Strategy score.
6. **Prize/legal**: any constraints on Kaggle competition eligibility for you
   (country, employer IP policy)?
