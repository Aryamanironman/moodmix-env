# DJSet-v1 - what to listen to (v2, vocal-aware)

All files are 44.1 kHz stereo WAV. The model is `models/ppo_v2_final.zip`,
the 1M-step run from the quiet clone (41-float observation, vocal terms).

## The set (1 min 51 s, six transitions)

`runs/v1/showcase_ppo_v2_final_set.wav`

| # | deck A to deck B | length | beat hits | level p99 jump | level range |
|---|---|---|---|---|---|
| 1 | Closer to Starboy | 20.2 s | 0.82 | 6.6 dB | 6.7 dB |
| 2 | STAY to Yellow | 22.6 s | 0.97 | 5.1 dB | 8.0 dB |
| 3 | BIRDS OF A FEATHER to Riptide | 18.3 s | 1.00 | 3.3 dB | 3.6 dB |
| 4 | Shape of You to Sweater Weather | 14.7 s | 0.85 | 3.4 dB | 4.1 dB |
| 5 | Blinding Lights to The Night We Met | 22.5 s | 0.94 | 2.6 dB | 2.7 dB |
| 6 | Believer to Sweater Weather | 15.4 s | 0.88 | 4.3 dB | 5.8 dB |

"Beat hits" is the fraction of the transition's beat grid with an onset peak
within 70 ms; "level p99 jump" is the 99th-percentile change in 250 ms RMS,
measured 0.5 s inside the edges. Both decks are phrase-cued and beat-warped to
deck A's grid.

## One pair, expert vs model

`runs/v1/v2/compare_side_by_side.wav` - Blinding Lights to The Night We Met,
expert first, 0.4 s gap, then the model.

- expert: +10.25, low overlap 0.293, mean level jump 0.289, beat hits 0.97
- model: +14.11, low overlap 0.100, mean level jump 0.245, beat hits 0.94

The two halves are `compare_expert.wav` and `compare_model.wav`.

## Stochastic samples

`runs/v1/v2/samples/sample_01.wav` to `sample_04.wav`, rewards +9.85, +11.22,
+11.29, +12.52. The exploration sigma is narrowed for the demo (log_std
-2.5); at the training sigma the samples are mostly noise.

## The models

- `models/ppo_v2_final.zip` - 1M PPO steps, eval +14.91, low overlap 0.116,
  mean level jump 0.171. Same weights as `models/ppo_v2.zip`.
- `models/ppo_v2_bc.zip` - behavior clone of the expert, 41-float obs, eval
  9.55 under the vocal-aware reward.
- `models/ppo_v2_bc_quiet.zip` - the clone with log_std -2.8, used as the PPO
  starting point.
- The 39-float v1 models are kept in `models/` for reference and do not load
  in the 41-float environment.

## Training records and lineage

- `runs/training/ppo_v2/` - `train_returns.csv`, `eval.csv`, `losses.csv`,
  `curves.png`, `report.html`, `report.md`.
- `mlops/` - the registry of every current and previous model and checkpoint,
  with SHA256 fingerprints, observation sizes, timesteps, eval metadata and
  copied records. The local player serves it at `/mlops/`.

## Your own songs

```bash
python scripts/prepare_library_v2.py --audio-dir "D:\Music" \
  --out "D:\Music\manifest_v2.json"
python scripts/mix_songs.py --data-dir "D:\Music" \
  --track-a "Song A.mp3" --track-b "Song B.mp3" \
  --model models/ppo_v2_final.zip --policy model --best-cues \
  --out runs/v1/my_blend.wav
```

Or run the local player (`python scripts/serve_v1.py --port 8050`), paste
the folder in the Library bar, and Load folder. The best-cues button searches
the strongest phrase pairs by the environment score; for a known pair, choose
the cues by ear, since the score is a proxy for the rendered audio.
