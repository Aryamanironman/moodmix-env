# DJSet-v1 - what to listen to

All files are 44.1 kHz stereo WAV. Start with the set, then the same pair as
expert vs model, then the stochastic range.

## The set (1 min 51 s, six transitions)

`showcase_ppo_v1_long_set.wav` - final model, `models/ppo_v1_final.zip`
(800k PPO steps from a behavior clone of the phrase-aware expert).

| # | deck A → deck B | length | beat hits | level p99 jump | level range |
|---|---|---|---|---|---|
| 1 | Closer → Starboy | 20.2 s | 0.91 | 3.7 dB | 4.6 dB |
| 2 | STAY → Yellow | 22.6 s | 1.00 | 4.7 dB | 6.3 dB |
| 3 | BIRDS OF A FEATHER → Riptide | 18.3 s | 0.97 | 3.5 dB | 3.7 dB |
| 4 | Shape of You → Sweater Weather | 14.7 s | 0.85 | 3.3 dB | 4.1 dB |
| 5 | Blinding Lights → The Night We Met | 22.5 s | 0.94 | 2.3 dB | 2.3 dB |
| 6 | Believer → Sweater Weather | 15.4 s | 0.88 | 4.8 dB | 6.3 dB |

"Beat hits" is the fraction of the transition's beat grid with an onset peak
within 70 ms; "level p99 jump" is the 99th-percentile change in 250 ms RMS,
measured 0.5 s inside the edges. Both decks are phrase-cued and beat-warped to
deck A's grid, so the beat score measures the render, not the tracker.

The expert reference of the same six is `showcase_expert_set.wav`, and the
behavior clone is `showcase_ppo_v1_bc_set.wav`.

## One pair, expert vs model

- `compare_side_by_side.wav` - expert first, 0.4 s gap, then the model.
  Riptide → Someone You Loved: expert +10.71, model +13.53; low-band overlap
  0.263 vs 0.155; mean level jump 0.276 vs 0.082; beat hits 0.97 vs 0.94.
- `compare_expert.wav` and `compare_model.wav` - the two halves.

## Stochastic samples (same pair, not argmax)

`sample_01.wav` … `sample_04.wav`, labelled in `samples.json`. Rewards
+8.73, +9.00, +8.12, +10.12 - a distribution of believable performances, with
the exploration sigma narrowed for the demo (`log_std = -2.5`).

## The models

- `models/ppo_v1_final.zip` - final: 800k PPO steps from the quiet clone, eval
  return 13.65 ± 0.41, low overlap 0.151, mean level jump 0.087. Same weights
  as `models/ppo_v1_long.zip`.
- `models/ppo_v1.zip` - 400k checkpoint (eval 12.99).
- `models/ppo_v1_bc.zip` - behavior clone of the expert: 39-float relative
  observation, 8 continuous controls, 400 expert episodes, MSE 4e-5, eval
  10.37.
- `models/ppo_v1_bc_quiet.zip` - same clone with `log_std = -2.8`, the
  starting point for fine-tuning. A clone with the original 0.22 sigma makes
  PPO learn a noise-robust 3-beat cut; the quiet clone keeps the blend.

## Your own songs

```bash
python scripts/prepare_library_v2.py --audio-dir "D:\Music" \
  --out "D:\Music\manifest_v2.json"
python scripts/mix_songs.py --data-dir "D:\Music" \
  --track-a "Song A.mp3" --track-b "Song B.mp3" \
  --model models/ppo_v1_final.zip --policy model --best-cues \
  --out runs/v1/my_blend.wav
```

The policy sees only relative features, so the same checkpoint applies to a
new folder. `--best-cues` tries the strongest phrase pairs; leave it off for a
random phrase.
