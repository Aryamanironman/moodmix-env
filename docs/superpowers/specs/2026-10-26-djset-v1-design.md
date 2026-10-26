# DJSet-v1 - smooth, generalisable DJ transitions

Date: 2026-10-26
Status: approved by the 5-hour build mandate; v0 remains intact

## Intent

The v0 agent scored well but the rendered mixes did not sound like a DJ. The
goal is a transition that sounds like one track: beat-locked, phrase-aligned,
bass-swapped, loudness-continuous, and reusable on any analysed songs.

## Why v0 failed (measured in recon)

1. The manifest stored no beat times. The renderer assumed beat 0 at t=0 and a
   constant `60/BPM`; Blinding Lights' first detected beat is 0.63 s in, and
   older analyses put it at 19.7 s.
2. Tempo folding changed the BPM number without folding the beat grid. STAY was
   tracked at 172 BPM (86 musical) while its grid stayed at 172, so a compatible
   pair asked for a 2× stretch.
3. The renderer's stretch ratio was inverted (`bpm_b / bpm_a`).
4. Ten discrete action bins made smooth automation impossible, and the reward
   paid for following a scripted fader curve rather than for a good mix.
5. Cues were arbitrary beat 0 / beat 32, not phrase boundaries, and deck B was
   not aligned to A's beats sample-accurately.
6. The mix had one-pole EQ, no loudness match, and no continuity penalty.

## Goals

- A 32-beat transition that is beat- and bar-aligned, with one bass swap and a
  filter release, that sounds like a hand-mixed blend.
- A policy trained on the 20-song crate that generalises to a new analysed
  folder through relative features only.
- Keep v0 env, tests and checkpoints working.

## Non-goals

- Stem separation, vocal extraction, looping/editing into a produced remix.
- Real-time streaming; rendering is offline.
- Replacing the v0 published numbers.

## Design

**Analysis (`library_v2.py`, `manifest_v2.json`)**
Real beat times, downbeat phase, per-beat low/mid/high/RMS/onset/percussive/
harmonic/centroid, bar energy/section, phrase cue points. Tempo octaves are
folded by subsampling the beat grid (choosing the offset with the strongest
onsets), not by changing the number alone.

**Environment (`env_v1.py`, `MoodMix/DJSet-v1`)**
Continuous actions: fader, EQ A low/mid/high, EQ B low/mid/high, filter. Decks
are beatmatched automatically. 39-float observation of relative features:
progress, fader/filter/EQ state, 12 features per deck (bands, RMS, onset,
percussive, harmonic, centroid, bar energy, downbeat, bar/phrase position),
mix overlap, tempo ratio, key distance. Reward: bass-swap exclusivity,
EQ separation gain, filter direction, phrase timing, fader jitter/route,
level and spectral continuity, low-band mud, terminal handover and mix quality.

**Rendering (`render_v2.py`)**
Beat-indexed placement, `rate = median(B beat) / median(A beat)`, per-beat warp
to A's grid, LR4 three-way crossover, TPT state-variable filter with a smooth
cutoff curve, loudness matching of B to A, equal-power fader, soft limiter,
stereo 44.1 kHz.

**Policy (`expert_v1.py`, `pretrain_v1.py`, `train_v1.py`)**
Phrase-aware scripted expert (fader arc, one bass swap, high-pass release) is
cloned with MSE on the continuous action mean, then PPO fine-tunes from the
clone. The expert is also the renderable reference.

**Demos (`demo_v1.py`, `mix_songs.py`)**
Any-song entry point, samples, set, and expert-vs-model comparison.

## Success measures

- Rendered beat-hit fraction at the transition's beat grid.
- p99 RMS jump (dB) across the mix.
- Reward gap between expert and a parked fader.
- Subjective listening by the user; the expert render is the floor.

## Results (2026-10-26)

- Expert reference: eval return 10.7-11.1 across the six showcase pairs.
- Behavior clone: eval 10.37, MSE 4e-5.
- PPO from a quiet clone (`log_std = -2.8`), 800k steps: eval **13.65 ±
  0.41**, low overlap 0.151, mean level jump 0.087. On the comparison pair it
  scores 13.53 vs the expert's 10.71, with 0.155 vs 0.263 low overlap and
  0.082 vs 0.276 mean level jump.
- Rendered showcase (six transitions, 111 s): beat hits 0.85-1.00, 250 ms RMS
  p99 jump 2.3-4.8 dB, level range 2.3-6.3 dB.
- Unseen pair (As It Was → The Night We Met): reward 12.85, beat hits 0.94,
  p99 4.5 dB.

Two reward games were found and closed by measurement:

1. A parked fader collected continuity points for a flat curve; continuity is
   now scaled by blend amount and mix quality requires blend time.
2. PPO learned a 3-beat cut to avoid simultaneous playback; a weak fader-arc
   term, a hard-cut charge, and a blend gate now make the blend strictly
   better.

And one optimisation trap: behavior cloning with a wide exploration sigma
(0.22) makes every PPO rollout a broken blend, so the policy learns the
noise-robust cut. Clone with `log_std = -2.8` first; then PPO preserves and
improves the blend.

## Update: vocals, folder scanning, records

The second pass adds:

- a per-beat **vocal likelihood** (harmonic fraction of 200-4000 Hz, low
  spectral flatness, vocal-range centroid), with `.lrc` timings taking
  precedence when present;
- **41-float observation** (vocal is the ninth per-deck feature);
- reward terms `vocal_clash` and `vocal_space`, plus vocal-shaped cue choice;
- renderer **ducking**: when both leads are present both mid/high bands give
  way, when one is present the other deck ducks under it;
- a **Library bar** in the local player that scans any folder, analyses it
  once, caches the manifest under `runs/v1/library_cache`, and swaps the
  dropdowns without writing to the user's music folder;
- a **TrainingLogger** that writes episode returns, deterministic eval
  metrics, losses, `curves.png`, `report.html` and `report.md` under
  `runs/training/`, plus `/training` and `/explainer` pages in the player.

The Vercel static showcase was removed. The demo is the local PC player on
`:8050`; commercial audio stays local.

## Constraints

Five hours from 2026-10-26, CPU only, no new dependencies beyond installed
librosa/scipy/numba/SB3. Memory is tight; training runs use 4-8 envs and
checkpoint so they can resume.
