"""Serve the browser demo.

    python scripts/serve_demo.py
    python scripts/serve_demo.py --model models/ppo_1m.zip --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodmix_env.webapp import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "ppo_100k.zip")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="folder containing manifest.json (default: demo crate)")
    args = parser.parse_args()

    app = create_app(model_path=args.model, data_dir=args.data_dir)
    print(f"MoodMix demo: http://{args.host}:{args.port}")
    print(f"model: {args.model if args.model.exists() else 'not found (PPO falls back to human)'}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
