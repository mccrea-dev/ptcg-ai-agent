"""Pure-numpy inference for the trained option ranker.

Mirrors ptcg.training.torch_model.OptionRanker.forward exactly, so the
submission bundle needs numpy only. Used first as the search agent's root
ordering (which options deserve rollouts), later as a rollout policy.
"""

import json
from pathlib import Path

import numpy as np

from ptcg.training.encoding import (
    N_OPTION_TYPES, N_SELECT_TYPES, N_CONTEXTS, PAD,
    board_features, board_card_ids,
)


def _relu(x):
    return np.maximum(x, 0.0)


class NumpyRanker:
    def __init__(self, npz_path: str | Path):
        data = np.load(npz_path)
        self.p = {k: data[k] for k in data.files if k != "vocab_json"}
        self.vocab = {int(k): v for k, v in
                      json.loads(bytes(data["vocab_json"]).decode()).items()}

    def _linear(self, x, prefix):
        return x @ self.p[f"{prefix}.weight"].T + self.p[f"{prefix}.bias"]

    def _mlp(self, x, prefix):
        return self._linear(_relu(self._linear(x, f"{prefix}.0")), f"{prefix}.2")

    def scores(self, select, board, attack_damage: dict) -> np.ndarray:
        """Score every option in a selection; higher = more expert-like."""
        options = select["option"] if isinstance(select, dict) else select.option

        def g(o, k):
            return o.get(k) if isinstance(o, dict) else getattr(o, k, None)

        emb = self.p["card_emb.weight"]
        pooled = emb[board_card_ids(board, self.vocab)].mean(axis=0)
        sel_type = min((select.get("type") if isinstance(select, dict)
                        else select.type) or 0, N_SELECT_TYPES - 1)
        sel_ctx = min((select.get("context") if isinstance(select, dict)
                       else select.context) or 0, N_CONTEXTS - 1)
        ctx_in = np.concatenate([
            pooled, np.array(board_features(board), dtype=np.float32),
            self.p["sel_type_emb.weight"][sel_type],
            self.p["ctx_emb.weight"][sel_ctx],
        ])
        ctx = self._mlp(ctx_in, "context_mlp")

        out = np.empty(len(options), dtype=np.float32)
        for i, o in enumerate(options):
            card_idx = self.vocab.get(g(o, "cardId") or 0, PAD)
            opt_type = min(g(o, "type") or 0, N_OPTION_TYPES - 1)
            attack_id = g(o, "attackId")
            opt_in = np.concatenate([
                emb[card_idx], self.p["opt_type_emb.weight"][opt_type],
                np.array([1.0 if attack_id else 0.0,
                          attack_damage.get(attack_id, 0) / 300.0], dtype=np.float32),
            ])
            opt_vec = self._mlp(opt_in, "option_mlp")
            out[i] = self._mlp(np.concatenate([opt_vec, ctx]), "scorer")[0]
        return out
