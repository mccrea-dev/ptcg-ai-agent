"""Build and verify the Kaggle submission bundle.

Stages main.py + deck.csv + the official cg/ SDK into dist/submission/,
plays one verification game against the built-in random agent, then packs
dist/submission.tar.gz. A bundle that crashes on Kaggle is an automatic loss,
so a build that hasn't played a full game refuses to produce an archive.

Usage: uv run python -m ptcg.submission.build [deck-name]
deck-name is a CSV under ptcg/decks/ (default: starter).
"""

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).parents[2]
STAGE = REPO / "dist" / "submission"
OFFICIAL_CG = REPO / "data" / "official" / "sample_submission" / "sample_submission" / "cg"

VERIFY_SCRIPT = """
import main
assert main._search is not None, "search agent failed to load; bundle would play random"
from kaggle_environments import make
env = make("cabt")
env.run(["main.py", "random"])
last = env.steps[-1]
assert last[0].status == "DONE" and last[1].status == "DONE", (last[0].status, last[1].status)
print("verify ok:", len(env.steps), "steps, reward", last[0].reward)
"""


def build(deck_name: str = "starter") -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    shutil.copy(REPO / "ptcg" / "submission" / "main.py", STAGE / "main.py")
    shutil.copy(REPO / "ptcg" / "decks" / f"{deck_name}.csv", STAGE / "deck.csv")
    shutil.copytree(OFFICIAL_CG, STAGE / "cg",
                    ignore=shutil.ignore_patterns("__pycache__"))
    # The agent imports the ptcg package, so ship it alongside main.py.
    shutil.copytree(REPO / "ptcg", STAGE / "ptcg",
                    ignore=shutil.ignore_patterns("__pycache__"))

    # cwd=STAGE so main.py resolves deck.csv the same way it will on Kaggle.
    # Small search budget keeps the verification game fast; on Kaggle the
    # budget comes from the real clock instead.
    env = dict(os.environ, PTCG_BUDGET_S="0.2")
    subprocess.run([sys.executable, "-c", VERIFY_SCRIPT], cwd=STAGE, check=True, env=env)

    archive = REPO / "dist" / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in STAGE.rglob("*"):
            tar.add(path, arcname=path.relative_to(STAGE))
    print("built", archive)
    return archive


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "starter")
