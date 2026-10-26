"""Analyze a folder into the beat-accurate v2 manifest used by DJSet-v1.

    python scripts/prepare_library_v2.py --audio-dir crate --out crate/manifest_v2.json
    python scripts/prepare_library_v2.py --audio-dir "D:\\Music" --out "D:\\Music\\manifest_v2.json"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodmix_env.library_v2 import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <audio-dir>/manifest_v2.json")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--mood", default=None,
                        help="force one mood for every track")
    args = parser.parse_args()

    audio_dir = args.audio_dir.resolve()
    out = (args.out or audio_dir / "manifest_v2.json").resolve()
    moods = None
    if args.mood:
        moods = {path.stem: args.mood for path in audio_dir.iterdir()
                 if path.is_file()}
    manifest = build_manifest(audio_dir, out, moods=moods,
                              sr=args.sample_rate)
    print(f"\nmanifest: {out}")
    print(f"tracks:   {len(manifest['tracks'])}")
    for track in manifest["tracks"]:
        cues = len(track["cues"])
        print(f"  {track['file'][:36]:36s} {track['bpm']:7.2f} BPM  "
              f"{track['key']:>2} {track['mode']:5s}  "
              f"{len(track['beat_times']):4d} beats  {cues:2d} cues  "
              f"{track['mood']}")


if __name__ == "__main__":
    main()
