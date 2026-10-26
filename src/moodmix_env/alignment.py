"""Automatic mix-to-track alignment.

Recovering DJ controls needs to know which part of each source track is
playing at every moment of the mix.  This module finds that mapping with
subsequence DTW on chroma features, which tolerates tempo differences and does
not need the operator to supply cue points.

The result is a linear map from mix time to source time::

    source_frame = slope * mix_frame + intercept

`slope` is the speed the track was played at (1.0 = original tempo), and
`intercept` locates the entry point.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

ALIGN_HOP = 2048


@dataclass
class Alignment:
    slope: float
    intercept: float
    hop: int
    score: float

    def source_frame(self, mix_frame: float) -> float:
        """Source frame (in this alignment's hop units) for a mix frame."""
        return self.slope * float(mix_frame) + self.intercept

    def source_seconds(self, mix_seconds: float, sr: int) -> float:
        frame = self.source_frame(mix_seconds * sr / self.hop)
        return frame * self.hop / sr


def chroma(y: np.ndarray, sr: int, hop: int = ALIGN_HOP) -> np.ndarray:
    """Pitch-class features with each frame L2-normalised for cosine DTW."""
    feats = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    # Digital silence (a track that has not been brought in yet, a gap in a
    # recording) normalises to an all-zero column, whose cosine distance is NaN
    # and poisons the whole DTW cost matrix.  A flat pitch-class vector is the
    # neutral frame: it matches every key weakly instead of crashing.
    silent = feats.sum(axis=0) <= 1e-9
    if np.any(silent):
        feats = feats.copy()
        feats[:, silent] = 1.0
    norms = np.linalg.norm(feats, axis=0, keepdims=True)
    return feats / (norms + 1e-9)


def align(track: np.ndarray, mix: np.ndarray, sr: int = 22050,
          hop: int = ALIGN_HOP, focus: np.ndarray | None = None,
          rate: float | None = None) -> Alignment:
    """Align a source track against a mix that contains it.

    `focus` is an optional boolean mask over mix frames to score the entry
    point on.  Pass the beats where this deck dominates: otherwise a track
    that is only audible early on can be matched to unrelated loud material.

    `rate` pins the playback speed instead of estimating it from the path.
    Chroma cannot resolve a few percent of pitch shift over a short clip, so
    when the source tempo is known (a synced deck plays at mix_tempo / source
    tempo) that beats guessing.
    """
    track_feats = chroma(track, sr, hop)
    mix_feats = chroma(mix, sr, hop)
    if track_feats.shape[1] < 4 or mix_feats.shape[1] < 4:
        return Alignment(1.0, 0.0, hop, float("nan"))

    cost, path = librosa.sequence.dtw(
        X=track_feats, Y=mix_feats, subseq=True, metric="cosine"
    )
    steps = np.array(path)[::-1]          # time order: [track_idx, mix_idx]
    track_idx, mix_idx = steps[0], steps[1]
    if rate is not None and rate > 0:
        slope = float(rate)
        intercept = float(np.median(track_idx - slope * mix_idx))
    elif len(mix_idx) < 4 or np.ptp(mix_idx) < 2:
        slope, intercept = 1.0, float(mix_idx[0]) if len(mix_idx) else 0.0
    else:
        fitted, _ = np.polyfit(mix_idx, track_idx, 1)
        if not np.isfinite(fitted) or fitted <= 0.05:
            fitted = 1.0
        # DJs rarely shift tempo more than ~10%, and an unconstrained fit on a
        # short or partly silent part of a mix can run away.  Clamp the rate,
        # then take the median residual so a few bad DTW steps (crowd noise, a
        # breakdown) cannot drag the entry point.
        slope = float(np.clip(fitted, 0.9, 1.1))
        intercept = float(np.median(track_idx - slope * mix_idx))
    intercept = _refine_offset(track_feats, mix_feats, slope, intercept, focus)
    normalised = float(cost[-1, -1] / max(1, len(mix_idx)))
    return Alignment(float(slope), float(intercept), hop, normalised)


def _resize_focus(focus: np.ndarray | None, n_frames: int) -> np.ndarray | None:
    """Fit a boolean mask to the feature frame count.

    Callers build masks from sample counts (``len(y) // hop``), which lands one
    frame either side of the chroma matrix.  Silently dropping a mismatched
    mask is expensive: the correlation search then falls back to every loud
    frame in the clip and can lock onto an unrelated section of a repetitive
    track.
    """
    if focus is None:
        return None
    mask = np.asarray(focus, dtype=bool).ravel()
    if len(mask) == n_frames:
        return mask
    if len(mask) > n_frames:
        return mask[:n_frames]
    return np.pad(mask, (0, n_frames - len(mask)), constant_values=False)


def _focus_columns(mix_feats: np.ndarray, focus: np.ndarray | None
                   ) -> tuple[np.ndarray, np.ndarray] | None:
    """Mix frames worth scoring, transposed for fast correlation."""
    focus = _resize_focus(focus, mix_feats.shape[1])
    if focus is not None:
        usable = focus.copy()
    else:
        # Score the frames that actually carry pitch information.  Energy is
        # the obvious choice, but silent frames are normalised to a flat
        # vector for DTW and therefore keep a unit norm; their pitch-class
        # spread is zero, which is what separates them from music.
        spread = mix_feats.std(axis=0)
        if not np.any(spread > 1e-6):
            return None
        usable = (spread >= np.median(spread)) & (spread > 1e-6)
    cols = np.flatnonzero(usable)
    if len(cols) < 4:
        return None
    return cols, mix_feats[:, cols]


def _lag_score(track_feats: np.ndarray, cols: np.ndarray, sel: np.ndarray,
               slope: float, lag: float) -> float:
    """Mean chroma dot product of the mix frames against the source at a lag."""
    idx = np.round(slope * cols + lag).astype(int)
    ok = (idx >= 0) & (idx < track_feats.shape[1])
    if ok.sum() < max(4, 0.3 * len(cols)):
        return -np.inf
    return float(np.mean(np.sum(track_feats[:, idx[ok]] * sel[:, ok], axis=0)))


def _refine_offset(track_feats: np.ndarray, mix_feats: np.ndarray,
                   slope: float, seed: float,
                   focus: np.ndarray | None = None) -> float:
    """Pick the lag whose chroma lines up best with the loud part of the mix.

    DTW can lock onto a repeated chorus and shift the whole path; a direct
    correlation search around plausible lags fixes that.  Only frames with
    pitch-class contrast are scored, so silence cannot win.
    """
    found = _focus_columns(mix_feats, focus)
    if found is None:
        return seed
    mix_cols, mix_sel = found

    def score(lag: float) -> float:
        return _lag_score(track_feats, mix_cols, mix_sel, slope, lag)

    span = track_feats.shape[1]
    coarse = range(int(-0.1 * span), max(1, int(span - 0.05 * len(mix_cols))), 4)
    best = seed
    best_score = score(seed)
    for lag in coarse:
        value = score(lag)
        if value > best_score:
            best, best_score = float(lag), value
    for lag in np.arange(best - 4, best + 4.01, 1.0):
        value = score(lag)
        if value > best_score:
            best, best_score = float(lag), value
    return float(best)


def refine_seconds(track: np.ndarray, mix: np.ndarray, sr: int,
                   slope: float, coarse_seconds: float,
                   hop: int = 512, window: float = 2.0,
                   focus: np.ndarray | None = None) -> float:
    """Fine-tune an entry point to within one short frame.

    The coarse DTW alignment works in ~93 ms frames, which is plenty for the
    fader but shifts every band comparison slightly.  Searching a small window
    at ~23 ms resolution recovers the sample-level offset.
    """
    track_feats = chroma(track, sr, hop)
    mix_feats = chroma(mix, sr, hop)
    focus = _resize_focus(focus, mix_feats.shape[1])
    if focus is None:
        spread = mix_feats.std(axis=0)
        if not np.any(spread > 1e-6):
            return coarse_seconds
        focus = (spread >= np.median(spread)) & (spread > 1e-6)
    cols = np.flatnonzero(focus)
    if len(cols) < 4:
        return coarse_seconds
    sel = mix_feats[:, cols]
    centre = coarse_seconds * sr / hop
    radius = window * sr / hop

    def score(lag: float) -> float:
        idx = np.round(slope * cols + lag).astype(int)
        ok = (idx >= 0) & (idx < track_feats.shape[1])
        if ok.sum() < max(4, 0.3 * len(cols)):
            return -np.inf
        return float(np.mean(np.sum(track_feats[:, idx[ok]] * sel[:, ok], axis=0)))

    best_lag, best_score = centre, score(centre)
    for lag in np.arange(centre - radius, centre + radius + 1.0, 1.0):
        value = score(lag)
        if value > best_score:
            best_lag, best_score = float(lag), value
    return float(best_lag * hop / sr)


def align_files(mix_path, track_path, sr: int = 22050) -> Alignment:
    mix, _ = librosa.load(mix_path, sr=sr, mono=True)
    track, _ = librosa.load(track_path, sr=sr, mono=True)
    return align(track, mix, sr=sr)
