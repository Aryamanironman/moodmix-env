"""Analyze a folder of audio files into a MoodMix manifest.

    python scripts/analyze_library.py --audio-dir C:\\Music\\crate
    python scripts/analyze_library.py --audio-dir ./crate --mood party

Then point the environment at that folder (manifest.json sits inside it):
    MoodMixEnv(data_dir="./crate")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moodmix_env.library import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <audio-dir>/manifest.json")
    parser.add_argument("--mood", default=None,
                        help="force one mood for every track; default: infer")
    parser.add_argument("--sample-rate", type=int, default=22050)
    args = parser.parse_args()

    audio_dir = args.audio_dir.resolve()
    out = (args.out or audio_dir / "manifest.json").resolve()
    moods = None
    if args.mood:
        moods = {path.stem: args.mood for path in audio_dir.iterdir()
                 if path.is_file()}
    manifest = build_manifest(audio_dir, out, moods=moods,
                              sr=args.sample_rate)
    print(f"\nmanifest: {out}")
    print(f"tracks:   {len(manifest['tracks'])}")
    for track in manifest["tracks"]:
        print(f"  {track['file']:40s} {track['bpm']:6.1f} BPM  "
              f"{track['key']:>2} {track['mode']:5s}  {track['mood']}")


if __name__ == "__main__":
    main()
