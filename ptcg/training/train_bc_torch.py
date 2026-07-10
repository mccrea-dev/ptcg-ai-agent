"""Train the neural option ranker on mined ladder decisions.

Local smoke test:  uv run python -m ptcg.training.train_bc_torch data/train/decisions_v0.jsonl -o data/train/bc_net.pt --limit 4000 --epochs 1
GCP full run:      python -m ptcg.training.train_bc_torch decisions_v0.jsonl -o bc_net.pt --epochs 20
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ptcg.training.torch_model import OptionRanker, build_card_vocab, decision_to_tensors

MIN_RATING = 1000.0


class DecisionDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(items):
    max_o = max(len(x["opt_card"]) for x in items)
    max_p = max(len(x["pool_cards"]) for x in items)
    batch = {"mask": torch.zeros(len(items), max_o, dtype=torch.bool)}

    def pad(key, width, dims=1):
        shape = (len(items), width) if dims == 1 else (len(items), width, items[0][key].shape[-1])
        out = torch.zeros(shape, dtype=items[0][key].dtype)
        for j, x in enumerate(items):
            out[j, : len(x[key])] = x[key]
        return out

    batch["opt_card"] = pad("opt_card", max_o)
    batch["opt_type"] = pad("opt_type", max_o)
    batch["opt_scalars"] = pad("opt_scalars", max_o, dims=2)
    batch["pool_cards"] = pad("pool_cards", max_p)
    for j, x in enumerate(items):
        batch["mask"][j, : len(x["opt_card"])] = True
    for key in ("sel_type", "sel_ctx", "board", "target"):
        batch[key] = torch.stack([x[key] for x in items])
    return batch


def load(path: Path, vocab, limit=None):
    from ptcg.agents.priority_agent import ATTACK_DAMAGE
    rows, episodes = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["rating"] < MIN_RATING or len(r["chosen"]) != 1:
                continue
            options = r["select"]["option"] or []
            if len(options) < 2 or r["select"].get("maxCount") != 1:
                continue
            rows.append(decision_to_tensors(r, vocab, ATTACK_DAMAGE))
            episodes.append(r["episode"])
            if limit and len(rows) >= limit:
                break
    return rows, episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        # Some Kaggle workers hand out GPUs older than the torch build supports.
        if torch.cuda.get_device_capability(0) >= (7, 0):
            device = "cuda"
        else:
            print("GPU too old for this torch build; using CPU")
    vocab = build_card_vocab()
    rows, episodes = load(args.data, vocab, args.limit)
    val_eps = set(sorted(set(episodes))[::10])
    train = [r for r, e in zip(rows, episodes) if e not in val_eps]
    val = [r for r, e in zip(rows, episodes) if e in val_eps]
    print(f"device={device}, train {len(train)}, val {len(val)}")

    model = OptionRanker(n_cards=len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_dl = DataLoader(DecisionDataset(train), batch_size=args.batch,
                          shuffle=True, collate_fn=collate)
    val_dl = DataLoader(DecisionDataset(val), batch_size=args.batch, collate_fn=collate)

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        loss_sum = n = 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            loss = torch.nn.functional.cross_entropy(logits, batch["target"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(batch["target"])
            n += len(batch["target"])
        model.eval()
        hits = total = 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                hits += (model(batch).argmax(-1) == batch["target"]).sum().item()
                total += len(batch["target"])
        acc = hits / max(total, 1)
        print(f"epoch {epoch}: loss {loss_sum / n:.4f}, val top-1 {acc:.1%}")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "vocab": vocab}, args.out)
    print(f"best val top-1 {best:.1%}, saved {args.out}")


if __name__ == "__main__":
    main()
