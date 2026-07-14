"""Train the value net: board -> tanh win expectation, in (-1, 1).

Tiny by design (card-embedding mean + 2-layer MLP): inference must run in
microseconds on Kaggle CPU inside rollout scoring, and the numpy export in
ptcg/agents/learned.py must reproduce it exactly. Validation splits on game
boundaries (positions of one game share an outcome — row-level splits leak).

Usage:
  uv run python -m ptcg.training.train_value data/train/value_mined.npz data/train/value_selfplay.npz -o data/train/value_net
"""

import argparse
from pathlib import Path

import numpy as np
import torch

EMB_DIM = 48
HIDDEN = 64


class ValueNet(torch.nn.Module):
    def __init__(self, n_cards: int, n_feats: int):
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(n_cards, EMB_DIM, mode="mean", padding_idx=0)
        self.fc1 = torch.nn.Linear(EMB_DIM + n_feats, HIDDEN)
        self.fc2 = torch.nn.Linear(HIDDEN, 1)

    def forward(self, ids, feats):
        v = torch.cat([self.emb(ids), feats], dim=1)
        v = torch.relu(self.fc1(v))
        return torch.tanh(self.fc2(v)).squeeze(1)


def load(paths: list[Path]):
    ids, feats, y, w = [], [], [], []
    for p in paths:
        d = np.load(p)
        ids.append(d["ids"]); feats.append(d["feats"]); y.append(d["y"]); w.append(d["w"])
    return (np.concatenate(ids), np.concatenate(feats),
            np.concatenate(y), np.concatenate(w))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, nargs="+")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--holdout", type=float, default=0.1)
    args = parser.parse_args()

    ids, feats, y, w = load(args.data)
    n = len(y)
    n_val = int(n * args.holdout)  # tail block: games are stored contiguously
    tr = slice(0, n - n_val)
    va = slice(n - n_val, n)
    from ptcg.training.encoding import build_card_vocab
    n_cards = len(build_card_vocab()) + 1

    model = ValueNet(n_cards, feats.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    t_ids = torch.tensor(ids.astype(np.int64))
    t_feats = torch.tensor(feats)
    t_y = torch.tensor(y)
    t_w = torch.tensor(w)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n - n_val)
        total = 0.0
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            out = model(t_ids[tr][b], t_feats[tr][b])
            loss = (torch.nn.functional.huber_loss(
                out, t_y[tr][b], delta=0.5, reduction="none") * t_w[tr][b]).mean()
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        model.eval()
        with torch.no_grad():
            pv = model(t_ids[va], t_feats[va])
            acc = float(((pv > 0) == (t_y[va] > 0)).float().mean())
            corr = float(torch.corrcoef(torch.stack([pv, t_y[va]]))[0, 1])
        marker = ""
        if acc > best_acc:
            best_acc = acc
            marker = " *"
            torch.save(model.state_dict(), str(args.out) + ".pt")
            np.savez(str(args.out) + ".npz",
                     emb=model.emb.weight.detach().numpy(),
                     w1=model.fc1.weight.detach().numpy(),
                     b1=model.fc1.bias.detach().numpy(),
                     w2=model.fc2.weight.detach().numpy(),
                     b2=model.fc2.bias.detach().numpy())
        print(f"epoch {epoch}: train_loss {total / (n - n_val):.4f} "
              f"val_acc {acc:.3f} val_corr {corr:.3f}{marker}", flush=True)
    print(f"best val win-prediction accuracy: {best_acc:.3f} "
          f"(coin flip = 0.5; saved {args.out}.npz)")


if __name__ == "__main__":
    main()
