"""Two-stage candidate selection for multi-candidate generation.

Stage 1 (spectral gate, always on if ``librosa`` is available): reject noisy /
degenerate takes using spectral metrics — high spectral flatness (= noise), near
silence, or low spectral contrast (= little harmonic structure).

Stage 2 (CLAP ranking, optional): rank the survivors by CLAP text-audio similarity
to the prompt (adherence). If CLAP isn't installed, rank survivors by the Stage-1
heuristic score instead.

Everything degrades gracefully: with no librosa/CLAP at all, the first candidate is
returned. This keeps the dependency surface optional while giving a real quality
lift when the libraries are present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from src.config.config import MUSICGEN_MIN_CLEAN_S, MUSICGEN_NOISE_FLATNESS


@dataclass
class CandidateScore:
    index: int
    spectral_score: float          # higher = more musical / less noisy
    flatness: float                # whole-clip mean flatness
    rms: float
    passed_gate: bool
    spectral_flatness_slope: float = 0.0   # +ve = rotting toward noise over time
    tail_flatness: float = 0.0             # mean flatness of the last ~30%
    clean_seconds: float = 0.0             # time before noise onset (else full dur)
    degrades: bool = False
    clap_score: Optional[float] = None
    final_score: float = 0.0
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 1: spectral heuristics
# ---------------------------------------------------------------------------


def _spectral_metrics(wav: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Return (flatness, rms, contrast). Falls back to numpy if librosa absent."""
    wav = np.asarray(wav, dtype=np.float32)
    rms = float(np.sqrt(np.mean(wav**2))) if wav.size else 0.0
    try:
        import librosa

        S = np.abs(librosa.stft(wav, n_fft=1024, hop_length=512)) + 1e-9
        flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
        contrast = float(np.mean(librosa.feature.spectral_contrast(S=S, sr=sr)))
        return flatness, rms, contrast
    except Exception:
        # Numpy fallback: spectral flatness = geo-mean / arith-mean of power spectrum.
        if wav.size == 0:
            return 1.0, rms, 0.0
        spec = np.abs(np.fft.rfft(wav)) ** 2 + 1e-9
        gmean = float(np.exp(np.mean(np.log(spec))))
        amean = float(np.mean(spec))
        flatness = gmean / amean
        contrast = float(np.std(np.log(spec)))
        return flatness, rms, contrast


def windowed_flatness(wav: np.ndarray, sr: int, win_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-window spectral flatness over time → (window_start_times, flatness)."""
    wav = np.asarray(wav, dtype=np.float32)
    n = max(1, int(win_s * sr))
    if wav.size < n * 2:  # too short to window meaningfully
        fl, _, _ = _spectral_metrics(wav, sr)
        return np.array([0.0]), np.array([fl])
    times, flats = [], []
    for start in range(0, wav.size - n + 1, n):
        fl, _, _ = _spectral_metrics(wav[start : start + n], sr)
        times.append(start / sr)
        flats.append(fl)
    return np.asarray(times, dtype=np.float64), np.asarray(flats, dtype=np.float64)


def flatness_slope(times: np.ndarray, flats: np.ndarray) -> float:
    """Least-squares slope of flatness vs time. +ve = degrading toward noise."""
    if times.size < 2:
        return 0.0
    return float(np.polyfit(times, flats, 1)[0])


def clean_seconds(times: np.ndarray, flats: np.ndarray, total_dur: float,
                  threshold: float = MUSICGEN_NOISE_FLATNESS) -> float:
    """First time the clip sustainedly crosses the noise threshold (else full dur)."""
    if times.size == 0:
        return total_dur
    for i in range(flats.size):
        if flats[i] > threshold and (i + 1 >= flats.size or flats[i + 1] > threshold * 0.7):
            return float(times[i])
    return float(total_dur)


def trim_trailing_noise(
    wav: np.ndarray,
    sr: int,
    threshold: float = MUSICGEN_NOISE_FLATNESS,
    min_clean_s: float = MUSICGEN_MIN_CLEAN_S,
) -> tuple[np.ndarray, float, bool]:
    """Cut a noisy tail off a clip. Returns (wav, new_duration_s, trimmed?).

    Finds where windowed flatness sustainedly crosses ``threshold`` and trims there,
    with a short fade-out. Never trims below ``min_clean_s`` (if noise starts earlier,
    keep the full clip and let the caller flag it).
    """
    wav = np.asarray(wav, dtype=np.float32)
    total = float(wav.size / sr) if sr else 0.0
    if total <= min_clean_s:
        return wav, total, False
    times, flats = windowed_flatness(wav, sr)
    cut = clean_seconds(times, flats, total, threshold)
    if cut >= total - 0.5:  # no meaningful noisy tail
        return wav, total, False
    cut = max(cut, min_clean_s)
    if cut >= total - 0.5:
        return wav, total, False
    n = int(cut * sr)
    trimmed = wav[:n].copy()
    fade = min(int(0.05 * sr), trimmed.size)  # 50 ms fade-out to avoid a click
    if fade > 1:
        trimmed[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return trimmed, float(trimmed.size / sr), True


def _stage1_score(flatness: float, rms: float, contrast: float, slope: float = 0.0) -> float:
    """Higher = more tonal/structured AND stable over time.

    Clean MusicGen takes sit around flatness 0.0001–0.015 while noisy/degenerate takes
    sit around 0.05–0.15. A *rising* flatness slope (~0.01/s) means the take is rotting
    mid-clip even if its head is clean — so we add a stability term: slope ~0.01/s
    fully penalises, ~0.001/s (steady) does not.
    """
    tonal = 1.0 - min(1.0, flatness * 12.0)       # flatness ~0.08+ -> treated as noisy
    stability = 1.0 - min(1.0, max(0.0, slope) * 100.0)  # slope 0.01/s -> 0
    loud_ok = min(1.0, rms / 0.06)                # very quiet -> low score
    structure = min(1.0, max(0.0, contrast / 20.0))
    return float(0.45 * tonal + 0.35 * stability + 0.1 * loud_ok + 0.1 * structure)


# ---------------------------------------------------------------------------
# Stage 2: CLAP (optional)
# ---------------------------------------------------------------------------


class _ClapScorer:
    """Lazy CLAP text-audio similarity, via transformers. Optional."""

    _instance: Optional["_ClapScorer"] = None
    _unavailable = False

    def __init__(self) -> None:
        from transformers import ClapModel, ClapProcessor  # type: ignore

        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.model.eval()

    @classmethod
    def get(cls) -> Optional["_ClapScorer"]:
        if cls._unavailable:
            return None
        if cls._instance is None:
            try:
                cls._instance = cls()
            except Exception:
                cls._unavailable = True
                return None
        return cls._instance

    def similarity(self, prompt: str, wav: np.ndarray, sr: int) -> float:
        import librosa
        import torch

        wav48 = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=48000)
        with torch.no_grad():
            inputs = self.processor(
                text=[prompt], audios=[wav48], sampling_rate=48000,
                return_tensors="pt", padding=True,
            )
            out = self.model(**inputs)
            # logits_per_audio: [1,1]; use the raw similarity.
            return float(out.logits_per_audio.squeeze().item())


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def score_candidates(
    waveforms: Sequence[np.ndarray],
    sample_rate: int,
    prompt: str,
    use_clap: bool = True,
) -> List[CandidateScore]:
    """Score each candidate through the two-stage pipeline."""
    scores: List[CandidateScore] = []
    for i, wav in enumerate(waveforms):
        wav = np.asarray(wav, dtype=np.float32)
        flatness, rms, contrast = _spectral_metrics(wav, sample_rate)
        total_dur = float(wav.size / sample_rate) if sample_rate else 0.0

        times, win_flats = windowed_flatness(wav, sample_rate)
        slope = flatness_slope(times, win_flats)
        tail = float(np.mean(win_flats[max(0, int(len(win_flats) * 0.7)) :])) if win_flats.size else flatness
        clean_s = clean_seconds(times, win_flats, total_dur)
        head = float(np.mean(win_flats[: max(1, int(len(win_flats) * 0.3))])) if win_flats.size else flatness
        degrades = (slope > 0.004) or (tail > MUSICGEN_NOISE_FLATNESS and tail > head * 3)

        s1 = _stage1_score(flatness, rms, contrast, slope=slope)
        # Gate: drop takes that are noisy overall, near-silent, or noisy by the end.
        passed = (flatness < 0.06) and (rms > 0.01) and (tail < 0.08)
        scores.append(
            CandidateScore(
                index=i, spectral_score=s1, flatness=flatness, rms=rms,
                spectral_flatness_slope=slope, tail_flatness=tail,
                clean_seconds=clean_s, degrades=degrades,
                passed_gate=passed, final_score=s1,
                notes=[] if passed else ["failed spectral gate"],
            )
        )

    survivors = [s for s in scores if s.passed_gate] or scores  # never drop everything

    clap = _ClapScorer.get() if use_clap else None
    if clap is not None:
        for s in survivors:
            try:
                s.clap_score = clap.similarity(prompt, waveforms[s.index], sample_rate)
                s.final_score = s.clap_score
            except Exception:
                s.clap_score = None  # keep stage-1 final_score
    return scores


def select_best(
    waveforms: Sequence[np.ndarray],
    sample_rate: int,
    prompt: str,
    use_clap: bool = True,
) -> tuple[int, List[CandidateScore]]:
    """Return (best_index, all_scores). Best = highest final_score among gate survivors."""
    if not waveforms:
        raise ValueError("no candidates to score")
    if len(waveforms) == 1:
        s = score_candidates(waveforms, sample_rate, prompt, use_clap=False)
        return 0, s

    scores = score_candidates(waveforms, sample_rate, prompt, use_clap=use_clap)
    survivors = [s for s in scores if s.passed_gate] or scores
    best = max(survivors, key=lambda s: s.final_score)
    return best.index, scores
