"""Neural behavior-cloning model: card-embedding option ranker.

The linear baseline (bc_v0) plateaued at priority-policy level because its
features can't express card identity — "fetch Kadabra, not Poffin, on turn 2"
needs an embedding per card. This model embeds the option's card, the option
type, the selection context, and a pooled view of the board (our hand, both
sides' Pokémon), then scores each option against that context.

Runs on CPU for smoke tests; real training happens on a GCP GPU.
"""

import torch
import torch.nn as nn

from ptcg.training.encoding import (  # noqa: F401  (re-exported for callers)
    N_OPTION_TYPES, N_SELECT_TYPES, N_CONTEXTS, N_BOARD_FEATS, PAD,
    build_card_vocab, board_features, board_card_ids,
)


class OptionRanker(nn.Module):
    def __init__(self, n_cards: int, dim: int = 48):
        super().__init__()
        self.card_emb = nn.Embedding(n_cards + 1, dim, padding_idx=PAD)
        self.opt_type_emb = nn.Embedding(N_OPTION_TYPES, 16)
        self.sel_type_emb = nn.Embedding(N_SELECT_TYPES, 8)
        self.ctx_emb = nn.Embedding(N_CONTEXTS, 16)
        self.context_mlp = nn.Sequential(
            nn.Linear(dim + N_BOARD_FEATS + 8 + 16, 96), nn.ReLU(), nn.Linear(96, 64))
        self.option_mlp = nn.Sequential(
            nn.Linear(dim + 16 + 2, 96), nn.ReLU(), nn.Linear(96, 64))
        self.scorer = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, batch):
        # batch: opt_card [B,O], opt_type [B,O], opt_scalars [B,O,2], mask [B,O],
        #        sel_type [B], sel_ctx [B], board [B,F], pool_cards [B,P]
        pooled = self.card_emb(batch["pool_cards"]).mean(dim=1)
        ctx = self.context_mlp(torch.cat(
            [pooled, batch["board"], self.sel_type_emb(batch["sel_type"]),
             self.ctx_emb(batch["sel_ctx"])], dim=-1))          # [B,64]
        opt = self.option_mlp(torch.cat(
            [self.card_emb(batch["opt_card"]), self.opt_type_emb(batch["opt_type"]),
             batch["opt_scalars"]], dim=-1))                     # [B,O,64]
        ctx = ctx.unsqueeze(1).expand(-1, opt.shape[1], -1)
        logits = self.scorer(torch.cat([opt, ctx], dim=-1)).squeeze(-1)  # [B,O]
        return logits.masked_fill(~batch["mask"], -1e9)


def decision_to_tensors(record: dict, vocab: dict[int, int], attack_damage: dict):
    select, board = record["select"], record["board"]
    options = select["option"]
    opt_card = [vocab.get(o.get("cardId") or 0, PAD) for o in options]
    opt_type = [min(o.get("type") or 0, N_OPTION_TYPES - 1) for o in options]
    opt_scalars = [[1.0 if o.get("attackId") else 0.0,
                    attack_damage.get(o.get("attackId"), 0) / 300.0] for o in options]
    return {
        "opt_card": torch.tensor(opt_card),
        "opt_type": torch.tensor(opt_type),
        "opt_scalars": torch.tensor(opt_scalars),
        "sel_type": torch.tensor(min(select.get("type") or 0, N_SELECT_TYPES - 1)),
        "sel_ctx": torch.tensor(min(select.get("context") or 0, N_CONTEXTS - 1)),
        "board": torch.tensor(board_features(board)),
        "pool_cards": torch.tensor(board_card_ids(board, vocab)),
        "target": torch.tensor(record["chosen"][0]),
    }
