"""Serve the DJSet-v1 web player.

    python scripts/serve_v1.py
    python scripts/serve_v1.py --model models/ppo_v1_final.zip --port 8050
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodmix_env.webapp_v1 import DEFAULT_MODEL, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "crate")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    app = create_app(data_dir=args.data_dir, model_path=args.model)
    print(f"DJSet-v1 player: http://{args.host}:{args.port}", flush=True)
    print(f"model: {args.model}", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True,
            use_reloader=False)


if __name__ == "__main__":
    main()
