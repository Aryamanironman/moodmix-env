# MoodMix DJSet-v1 player - visual redesign spec

## What it is

A local desktop player for the DJSet-v1 model. The user points it at a music
folder, picks two tracks and phrase cues, renders a transition, plays it, and
watches the model's fader, EQ, filter, score and vocal-aware behaviour.

## Audience and scene

The owner and friends, at a laptop, showing the project. It should feel like
standing at a real DJ booth or looking over a DJ's shoulder, not like a
generic analytics dashboard. Phone width matters for sharing, but the main
scene is 1440 x 900 on a laptop.

## Core content (must keep)

- Library: current folder, folder path, Browse, Load folder, built-in crate.
- Deck A and B: track, BPM, key, mood, phrase cue, live section.
- Render / Model / Expert / best-cues controls.
- Play, seek, time.
- Crossfader, EQ A/B (low/mid/high), filter.
- Fader / target / level / low-overlap timeline.
- Episode score and component breakdown.
- Training (MLOps) and Explainer links.
- Real content: 20 crate tracks, Blinding Lights -> The Night We Met, score
  14.98, cues 67/130, 22.4 s, components from the v2 model.

## Tone

Tactile, loud, physical, precise. Club energy without losing readability.
Not a SaaS dashboard. No invented data, no fake logos, no brand imitation.

## Format

- Three full responsive HTML directions, one shared content set.
- 1440 x 900 and 375 x 812 must both work.
- Final implementation will be adapted into `web_v1/index.html` against the
  existing Flask API, keeping all current functions.

## Assumptions (no separate question round was needed)

- No brand system exists; use original interface design and typography.
- The user wants a stronger DJ identity, not a literal copy of Pioneer or
  Traktor. References inform principles, not trademarks.
- Audio stays local; design screens may reference the local server audio.
- Google Fonts with system fallbacks; no image generation available.

## Verified references

- Pioneer DJ CDJ-3000 official page: https://www.pioneerdj.com/en/product/dj-players-turntables/cdj-3000/
- Native Instruments Traktor Pro 4: https://www.native-instruments.com/products/traktor/dj-software/traktor-pro-4/
- Teenage Engineering TX-6: https://teenage.engineering/products/tx-6
- Reference boards: `design-demos/refs/dj-interfaces.jpg`,
  `design-demos/refs/te.jpg`
- Style specimens: huashu-design web #18 Gallery Dark,
  east #10 Hattori Playful Offset.

## Success

A first-time viewer reads it as a DJ tool in one second, can find the two
tracks and the crossfader without instruction, and the page has one memorable
visual idea that comes from the act of mixing.
