"""Download the official competition data into data/official/ (fresh machine
bootstrap). Skips the two ~150MB card-gallery PDFs; everything else — card
CSVs, engine source, the sample_submission SDK — lands with structure intact.

Usage: uv run python tools/fetch_official.py
Requires kaggle CLI auth and accepted competition rules.
"""

import csv
import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
DEST = ROOT / "data" / "official"
COMP = "pokemon-tcg-ai-battle"


def kaggle(*args) -> str:
    result = subprocess.run(["uv", "run", "kaggle", *args],
                            capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout


def main():
    listing = kaggle("competitions", "files", COMP, "--page-size", "500", "--csv")
    rows = [r for r in csv.DictReader(io.StringIO(listing))
            if r.get("name") and not r["name"].lower().endswith(".pdf")]
    print(f"{len(rows)} files to fetch")
    for r in rows:
        name = r["name"]
        target_dir = DEST / Path(name).parent
        target = DEST / name
        if target.exists():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        kaggle("competitions", "download", COMP, "-f", name,
               "-p", str(target_dir), "--quiet")
    # kaggle zips larger files; unzip any archives in place
    import zipfile
    for z in DEST.rglob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(z.parent)
        z.unlink()
    print("done ->", DEST)


if __name__ == "__main__":
    main()
