"""Access to the official competition SDK (the `cg` package).

Inside a Kaggle submission bundle `cg/` sits next to main.py and imports
directly. Locally it lives in the downloaded competition data, so we add that
directory to sys.path. Import `api` and `game` from here instead of importing
`cg` yourself; this module also avoids importing kaggle_environments, whose
startup costs ~10s.
"""

import sys
from pathlib import Path

_SDK_DIR = Path(__file__).parents[2] / "data" / "official" / "sample_submission" / "sample_submission"

try:
    import cg.api as api
    import cg.game as game
except ImportError:
    if not _SDK_DIR.exists():
        raise ImportError(
            f"Official SDK not found at {_SDK_DIR}. Download the competition data first "
            "(see README)."
        ) from None
    sys.path.insert(0, str(_SDK_DIR))
    import cg.api as api  # noqa: F401
    import cg.game as game  # noqa: F401
