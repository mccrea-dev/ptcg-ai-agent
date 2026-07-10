# Trains the PTCG behavior-cloning option ranker on Kaggle's free GPU.
# Inputs: the ptcg-train-data dataset (decisions + source zip) and the
# competition data (for the cg engine + card metadata).

import shutil
import sys
from pathlib import Path

INPUT = Path("/kaggle/input")
# Mount layout varies (flat vs datasets/<user>/... on newer images) — discover.
DATA = next(INPUT.rglob("ptcg-train-data"))
COMP = next(INPUT.rglob("pokemon-tcg-ai-battle"))
WORK = Path("/kaggle/working")

# Kaggle auto-extracts uploaded archives and silently drops empty files, so
# anchor on a real module, copy the tree somewhere writable, and recreate the
# (empty) __init__.py files the package needs.
import os
print("input root:", os.listdir("/kaggle/input") if Path("/kaggle/input").exists() else "MISSING")
for d in Path("/kaggle/input").iterdir():
    print(" ", d.name, "->", [p.name for p in list(d.iterdir())[:10]])

anchors = list(DATA.rglob("priority_agent.py"))
if not anchors:
    print("dataset tree:", [str(p) for p in list(DATA.rglob("*"))[:30]])
    raise FileNotFoundError("ptcg source not found in dataset")
src_pkg = anchors[0].parent.parent
work_pkg = WORK / "src" / "ptcg"
shutil.copytree(src_pkg, work_pkg, dirs_exist_ok=True)
for d in [work_pkg] + [p for p in work_pkg.rglob("*")
                       if p.is_dir() and "__pycache__" not in p.parts]:
    (d / "__init__.py").touch()
sys.path.insert(0, str(work_pkg.parent))

# The cg package ships with the competition data; ptcg.engine.sdk falls back
# to a repo path that doesn't exist here, so put cg on sys.path first.
sample = next(COMP.glob("**/sample_submission/cg")).parent
sys.path.insert(0, str(sample))

# decks CSVs live inside the extracted source, so ptcg imports work as-is.
sys.argv = ["train_bc_torch", str(DATA / "decisions_v0.jsonl"),
            "-o", str(WORK / "bc_net.pt"), "--epochs", "25"]
from ptcg.training.train_bc_torch import main

main()
print("done; output at", WORK / "bc_net.pt")
