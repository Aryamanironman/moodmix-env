# Direction approval

Date: 2026-10-26
Task: restyle the MoodMix player so it looks like an actual DJ set.

## What was shown

Six thumbnail candidates were drawn, then three full directions were built with
real crate data and the recorded Blinding Lights to The Night We Met transition:

| Option | Direction | Full draft | Desktop capture |
|---|---|---|---|
| A | Silver Booth - hardware club console | `design-demos/dir-1.html` | `design-demos/dir-1-1440.png` |
| B | Performance View - Traktor style software console | `design-demos/dir-2.html` | `design-demos/dir-2-1440.png` |
| C | Printed Scorecard - risograph sheet | `design-demos/dir-3.html` | `design-demos/dir-3-1440.png` |

The comparison sheet with all three screenshots is `design-demos/options.html`.
The six thumbnails are `design-demos/thumbs.png`.

## User choice

The user said:

> "clue booth is good lets do that"

That is the club booth, option A, Silver Booth. Direction A is approved for
implementation.

## Implementation

- Live player: `web_v1/index.html`
- Desktop capture of the live app: `design-demos/silver-live-1440.png`
- Playing state: `design-demos/silver-live-playing-1440.png`
- Everything from the previous functional player is still wired: library path,
  Browse, Load, Built-in crate, deck track and cue selects, Model / Expert,
  best cues, Render, Next pair, play, seek, audio, fader, EQ, filter, VU,
  chart, score components, MLOps and Explainer links, crate.
- Design lint: no red lines. Yellow warnings are the hardware panel micro-labels
  (uppercase tags, mono labels), which are part of the chosen console direction,
  not template decoration.
