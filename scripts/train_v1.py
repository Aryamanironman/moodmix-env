"""Entry point: python scripts/train_v1.py --help"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodmix_env.train_v1 import main

if __name__ == "__main__":
    main()
