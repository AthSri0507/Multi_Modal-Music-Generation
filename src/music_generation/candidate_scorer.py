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


@dataclass
class CandidateScore:
    index: int
    spectral_score: float          # higher = more musical / less noisy
    flatness: float
    rms: float
    passed_gate: bool
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


def _stage1_score(flatness: float, rms: float, contrast: float) -> float:
    """Higher = more tonal/structured. Penalise flat (noisy) spectra + near silence.

    Empirically, clean MusicGen takes sit around flatness 0.0001–0.015 while noisy /
    degenerate takes sit around 0.05–0.15 — far below pure white noise (~1.0). So we
    make the tonal term steep in that low range (``flatness * 12`` saturates by ~0.08)
    to strongly prefer the cleanest candidate of the batch.
    """
    tonal = 1.0 - min(1.0, flatness * 12.0)       # flatness ~0.08+ -> treated as noisy
    loud_ok = min(1.0, rms / 0.06)                # very quiet -> low score
    structure = min(1.0, max(0.0, contrast / 20.0))
    return float(0.7 * tonal + 0.15 * loud_ok + 0.15 * structure)


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
        flatness, rms, contrast = _spectral_metrics(wav, sample_rate)
        s1 = _stage1_score(flatness, rms, contrast)
        # Gate: drop noisy takes (flatness above the clean-music band ~0.02) and
        # near-silence. Loose enough to keep bright/percussive music, strict enough
        # to reject the 0.05–0.15 "hiss/noise" takes MusicGen-small sometimes emits.
        passed = (flatness < 0.06) and (rms > 0.01)
        scores.append(
            CandidateScore(
                index=i, spectral_score=s1, flatness=flatness, rms=rms,
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
