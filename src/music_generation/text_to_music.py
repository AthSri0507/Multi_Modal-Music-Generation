"""End-to-end text -> understanding -> rendered audio pipeline.

Shared by the CLI, the FastAPI endpoint, and the Spark batch job. Two prompt paths:

* **enhanced** (default): ``prompt_understanding.analyze_prompt`` →
  ``enhanced_prompt_builder.build`` — rich, taxonomy-grounded prompts with genre /
  purpose / atmosphere / instrumentation.
* **legacy**: ``keyword_extract_spec`` → ``MusicPromptBuilder`` (kept for A/B and
  backward compatibility).

Generation uses **presets** (sampling settings + candidate count). For non-Fast
presets we render N≥3 candidates and keep the best via the two-stage
``candidate_scorer`` (spectral gate → optional CLAP).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

from src.config.config import (
    MUSICGEN_AUTOTRIM,
    MUSICGEN_DEFAULT_DURATION_S,
    MUSICGEN_STABLE_TEMPERATURE,
    MUSICGEN_STABLE_TOP_K,
)
from src.music_generation.audio_generator import GeneratedAudio, MusicGenAudioGenerator
from src.music_generation.candidate_scorer import select_best, trim_trailing_noise
from src.music_generation.control_schema import MusicControlSpec
from src.music_generation.enhanced_prompt_builder import build as build_enhanced
from src.music_generation.mood_map import detect_mood, quadrant_to_mood
from src.music_generation.presets import (
    GenerationPreset,
    clamp_duration_for_preset,
    get_preset,
)
from src.music_generation.prompt_builder import MusicPromptBuilder, resolve_duration
from src.music_generation.prompt_to_control import PromptToControlMapper
from src.music_generation.prompt_understanding import MusicRequest, analyze_prompt


# Per-mood tempo / intensity defaults used by the legacy extractor.
_MOOD_DEFAULTS: Dict[str, tuple[float, float]] = {
    "happy": (125.0, 0.70),
    "calm": (70.0, 0.30),
    "sad": (72.0, 0.35),
    "tense": (130.0, 0.85),
    "romantic": (90.0, 0.45),
    "neutral": (100.0, 0.50),
}


def _argmax_if_positive(scores: Dict[str, float]) -> Optional[str]:
    if not scores:
        return None
    key, val = max(scores.items(), key=lambda kv: kv[1])
    return key if val > 0 else None


def keyword_extract_spec(prompt: str) -> MusicControlSpec:
    """Legacy: build a MusicControlSpec from prompt keywords."""
    M = PromptToControlMapper  # static helpers only; no model instantiated.

    mood = detect_mood(prompt)
    if mood is None:
        quadrant = _argmax_if_positive(M._keyword_bonus(prompt, "emotion"))
        mood = quadrant_to_mood(quadrant) if quadrant else "neutral"

    style = _argmax_if_positive(M._keyword_bonus(prompt, "style")) or "generic"
    instrumentation = _argmax_if_positive(M._keyword_bonus(prompt, "instrumentation")) or "ensemble"

    default_tempo, default_intensity = _MOOD_DEFAULTS.get(mood, (100.0, 0.5))
    tempo = M._extract_bpm(prompt) or M._keyword_tempo(prompt) or default_tempo
    intensity = M._keyword_intensity(prompt)
    if intensity is None:
        intensity = default_intensity

    return MusicControlSpec(
        emotion=mood, style=style, tempo_bpm=float(tempo),
        intensity=float(intensity), instrumentation=instrumentation,
    )


def _spec_from_request(req: MusicRequest) -> MusicControlSpec:
    """Synthesize a legacy MusicControlSpec from a MusicRequest (back-compat)."""
    return MusicControlSpec(
        emotion=req.mood,
        style=req.genre,
        tempo_bpm=req.tempo_bpm,
        intensity=req.intensity,
        instrumentation=(req.instrumentation[0] if req.instrumentation else "ensemble"),
    )


@dataclass
class GenerationPlan:
    """Resolved generation inputs (no audio yet)."""

    spec: MusicControlSpec
    music_prompt: str
    duration_s: float
    preset: str
    request: Optional[MusicRequest] = None


@dataclass
class GenerationResult:
    """Everything produced for one text-to-music request."""

    audio: GeneratedAudio
    spec: MusicControlSpec
    music_prompt: str
    duration_s: float
    raw_prompt: str
    request: Optional[MusicRequest] = None
    preset: str = "balanced"
    num_candidates: int = 1
    candidate_scores: List[dict] = field(default_factory=list)
    trimmed: bool = False
    clean_seconds: Optional[float] = None


class TextToMusicPipeline:
    """Reusable text -> rendered audio pipeline (lazy heavy components)."""

    def __init__(
        self,
        use_bert: bool = False,
        fast: bool = False,
        enhanced: bool = True,
        generator: Optional[MusicGenAudioGenerator] = None,
        prompt_builder: Optional[MusicPromptBuilder] = None,
    ) -> None:
        self.use_bert = use_bert
        self.fast = fast
        self.enhanced = enhanced
        self._generator = generator
        self._builder = prompt_builder or MusicPromptBuilder()
        self._bert_mapper: Optional[PromptToControlMapper] = None
        self._use_clap = os.environ.get("MUSICGEN_USE_CLAP") == "1"

    # -- lazy components -----------------------------------------------------

    @property
    def generator(self) -> MusicGenAudioGenerator:
        if self._generator is None:
            self._generator = MusicGenAudioGenerator(fast=self.fast)
        return self._generator

    def _extract_legacy(self, prompt: str) -> MusicControlSpec:
        if not self.use_bert:
            return keyword_extract_spec(prompt)
        if self._bert_mapper is None:
            from src.music_generation.control_schema import MusicControlSchema

            schema = MusicControlSchema(emotion_to_id={"q1": 0, "q2": 1, "q3": 2, "q4": 3})
            self._bert_mapper = PromptToControlMapper(schema)
        spec = self._bert_mapper.map_prompt(prompt).spec
        return replace(spec, emotion=quadrant_to_mood(spec.emotion))

    # -- planning ------------------------------------------------------------

    def _resolve_preset(self, preset: Optional[str], fast: Optional[bool]) -> GenerationPreset:
        if preset is not None:
            return get_preset(preset)
        if fast is not None:
            return get_preset("fast" if fast else "balanced")
        return get_preset("fast" if self.fast else None)

    def plan(
        self,
        prompt: str,
        duration_s: Optional[float] = None,
        preset: Optional[str] = None,
        fast: Optional[bool] = None,
    ) -> GenerationPlan:
        """Resolve prompt + controls + duration WITHOUT generating audio."""
        preset_obj = self._resolve_preset(preset, fast)

        if self.enhanced:
            req = analyze_prompt(prompt)
            if preset_obj.genre_bias and not req.genre_explicit:
                req = replace(req, genre=preset_obj.genre_bias)
            music_prompt = build_enhanced(req)
            if preset_obj.extra_descriptor:
                music_prompt = music_prompt.replace(
                    ". Instrumental only", f", {preset_obj.extra_descriptor}. Instrumental only"
                )
            spec = _spec_from_request(req)
            dur_req = duration_s if duration_s is not None else req.duration_s
            duration = resolve_duration(dur_req, prompt_text=prompt, default=MUSICGEN_DEFAULT_DURATION_S)
            duration = clamp_duration_for_preset(duration, preset_obj)
            return GenerationPlan(spec, music_prompt, duration, preset_obj.name, req)

        spec = self._extract_legacy(prompt)
        music_prompt = self._builder.build(spec, raw_prompt=prompt)
        duration = resolve_duration(duration_s, prompt_text=prompt, default=MUSICGEN_DEFAULT_DURATION_S)
        duration = clamp_duration_for_preset(duration, preset_obj)
        return GenerationPlan(spec, music_prompt, duration, preset_obj.name, None)

    # -- generation ----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        duration_s: Optional[float] = None,
        seed: Optional[int] = None,
        fast: Optional[bool] = None,
        preset: Optional[str] = None,
        num_candidates: Optional[int] = None,
    ) -> GenerationResult:
        """Run the full pipeline: plan → render N candidates → select best."""
        plan = self.plan(prompt, duration_s, preset=preset, fast=fast)
        p = get_preset(plan.preset)
        n = int(num_candidates) if num_candidates is not None else p.num_candidates
        n = max(1, n)

        # Stability-biased sampling for sparse/low-energy prompts (they drift fastest).
        temperature, top_k = p.temperature, p.top_k
        if self._is_sparse(plan.request):
            temperature = min(temperature, MUSICGEN_STABLE_TEMPERATURE)
            top_k = min(top_k, MUSICGEN_STABLE_TOP_K)

        candidates = self.generator.generate_candidates(
            plan.music_prompt,
            n=n,
            duration_s=plan.duration_s,
            base_seed=seed,
            guidance_scale=p.guidance_scale,
            temperature=temperature,
            top_k=top_k,
            top_p=p.top_p,
        )
        waves = [c.waveform for c in candidates]
        best_idx, scores = select_best(
            waves, candidates[0].sample_rate, plan.music_prompt, use_clap=self._use_clap
        )
        audio = candidates[best_idx]
        chosen = next(s for s in scores if s.index == best_idx)

        # Trim a noisy tail off the chosen take so we never ship the drift.
        trimmed = False
        clean_s = chosen.clean_seconds
        if MUSICGEN_AUTOTRIM:
            new_wav, new_dur, trimmed = trim_trailing_noise(audio.waveform, audio.sample_rate)
            if trimmed:
                audio = replace(audio, waveform=new_wav, duration_s=new_dur)

        score_dicts = [
            {
                "index": s.index,
                "spectral_score": round(s.spectral_score, 4),
                "flatness": round(s.flatness, 4),
                "spectral_flatness_slope": round(s.spectral_flatness_slope, 5),
                "clean_seconds": round(s.clean_seconds, 1),
                "degrades": s.degrades,
                "clap_score": (None if s.clap_score is None else round(s.clap_score, 4)),
                "passed_gate": s.passed_gate,
                "chosen": s.index == best_idx,
            }
            for s in scores
        ]

        return GenerationResult(
            audio=audio,
            spec=plan.spec,
            music_prompt=plan.music_prompt,
            duration_s=audio.duration_s,
            raw_prompt=prompt,
            request=plan.request,
            preset=plan.preset,
            num_candidates=n,
            candidate_scores=score_dicts,
            trimmed=trimmed,
            clean_seconds=round(clean_s, 1),
        )

    @staticmethod
    def _is_sparse(request: Optional[MusicRequest]) -> bool:
        if request is None:
            return False
        return request.energy == "low" or request.genre in {
            "ambient", "cinematic", "classical", "piano", "lofi",
        }

    def generate_to_file(
        self,
        prompt: str,
        out_path: str | Path,
        duration_s: Optional[float] = None,
        seed: Optional[int] = None,
        fast: Optional[bool] = None,
        preset: Optional[str] = None,
    ) -> GenerationResult:
        result = self.generate(prompt, duration_s=duration_s, seed=seed, fast=fast, preset=preset)
        MusicGenAudioGenerator.write_wav(result.audio, out_path)
        return result
