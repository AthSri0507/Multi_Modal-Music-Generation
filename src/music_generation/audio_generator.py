"""Text-to-audio generation with MusicGen.

Wraps the pretrained, instrumental ``facebook/musicgen-small`` model (via Hugging
Face Transformers) to render multi-instrument audio from a text prompt. MusicGen
does not produce vocals, which matches the project goal of "a tune, no lyrics".

The model is loaded lazily on first use. CPU is fully supported (the project's
default environment); generation time scales roughly linearly with the requested
duration -- expect a few minutes for ~10s of audio on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from src.config.config import (
    MUSICGEN_DEFAULT_DURATION_S,
    MUSICGEN_FAST_GUIDANCE_SCALE,
    MUSICGEN_GUIDANCE_SCALE,
    MUSICGEN_MAX_DURATION_S,
    MUSICGEN_MIN_DURATION_S,
    MUSICGEN_MODEL_NAME,
    MUSICGEN_TOKENS_PER_SECOND,
)


@dataclass
class GeneratedAudio:
    """A single rendered clip."""

    waveform: np.ndarray  # float32, shape [num_samples]
    sample_rate: int
    duration_s: float
    prompt: str


class MusicGenAudioGenerator:
    """Lazy-loading wrapper around a MusicGen text-to-music model."""

    def __init__(
        self,
        model_name: str = MUSICGEN_MODEL_NAME,
        device: str = "auto",
        guidance_scale: float = MUSICGEN_GUIDANCE_SCALE,
        quantize: bool = False,
        bf16: bool = False,
        num_threads: Optional[int] = None,
        fast: bool = False,
    ) -> None:
        # `fast` is a convenience: drop classifier-free guidance + int8-quantize.
        if fast:
            guidance_scale = MUSICGEN_FAST_GUIDANCE_SCALE
            quantize = True
        # int8 dynamic quant and bf16 autocast are mutually exclusive: a quantized
        # qlinear_dynamic kernel requires float (not bf16) input. Prefer quantize.
        if quantize and bf16:
            bf16 = False
        self.model_name = model_name
        self.guidance_scale = float(guidance_scale)
        self.quantize = bool(quantize)
        self.bf16 = bool(bf16)
        self.num_threads = num_threads
        self._device = device
        self._model = None
        self._processor = None
        self._sample_rate: Optional[int] = None

    # -- model loading -------------------------------------------------------

    def _resolve_device(self) -> str:
        if self._device != "auto":
            return self._device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "transformers>=4.31 is required for MusicGen. Run "
                "`pip install -r requirements.txt`."
            ) from exc

        import torch

        # Use all logical cores for the CPU matmuls unless told otherwise.
        if self.num_threads:
            torch.set_num_threads(int(self.num_threads))
        elif self._resolve_device() == "cpu":
            import os

            torch.set_num_threads(os.cpu_count() or torch.get_num_threads())

        device = self._resolve_device()
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(self.model_name)
        model.to(device)
        model.eval()

        # int8 dynamic quantization of the Linear layers — a meaningful CPU speedup
        # on Intel/oneDNN for this Linear-heavy transformer, at a small quality cost.
        if self.quantize and device == "cpu":
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )

        self._model = model
        self._sample_rate = int(self._model.config.audio_encoder.sampling_rate)

    @property
    def sample_rate(self) -> int:
        self._ensure_loaded()
        assert self._sample_rate is not None
        return self._sample_rate

    # -- generation ----------------------------------------------------------

    @staticmethod
    def _clamp_duration(duration_s: float) -> float:
        return float(min(MUSICGEN_MAX_DURATION_S, max(MUSICGEN_MIN_DURATION_S, duration_s)))

    def _max_new_tokens(self, duration_s: float) -> int:
        return int(round(duration_s * MUSICGEN_TOKENS_PER_SECOND))

    @staticmethod
    def _normalize(
        wav: np.ndarray,
        target_rms: float = 0.14,
        peak_ceiling: float = 0.95,
        max_gain: float = 6.0,
    ) -> np.ndarray:
        """Loudness-normalize to a consistent level without amplifying noise.

        Earlier we peak-normalized to a fixed level, which blew up quiet/noisy tails
        to full scale (making "noise" louder). Instead we target a constant RMS, but
        **cap the gain** so near-silent or noisy takes aren't boosted, and **limit
        peaks** to avoid clipping.
        """
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size == 0:
            return wav
        rms = float(np.sqrt(np.mean(wav**2)))
        if rms < 1e-5:  # essentially silence — leave it alone
            return wav
        gain = min(target_rms / rms, max_gain)
        wav = wav * gain
        peak = float(np.max(np.abs(wav)))
        if peak > peak_ceiling:
            wav = wav * (peak_ceiling / peak)
        return wav.astype(np.float32)

    def generate(
        self,
        prompt: str,
        duration_s: float = MUSICGEN_DEFAULT_DURATION_S,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> GeneratedAudio:
        """Render a single instrumental clip from a text prompt."""
        results = self.generate_batch(
            [prompt], duration_s=duration_s, seed=seed, guidance_scale=guidance_scale,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
        return results[0]

    def generate_batch(
        self,
        prompts: Sequence[str],
        duration_s: float = MUSICGEN_DEFAULT_DURATION_S,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> List[GeneratedAudio]:
        """Render one clip per prompt (all share the same duration)."""
        if not prompts:
            return []
        self._ensure_loaded()
        import torch

        duration_s = self._clamp_duration(duration_s)
        max_new_tokens = self._max_new_tokens(duration_s)
        guidance = self.guidance_scale if guidance_scale is None else float(guidance_scale)

        if seed is not None:
            torch.manual_seed(int(seed))

        device = self._resolve_device()
        inputs = self._processor(text=list(prompts), padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: dict = {
            "do_sample": True,
            "guidance_scale": guidance,
            "max_new_tokens": max_new_tokens,
        }
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        if top_k is not None and top_k > 0:
            gen_kwargs["top_k"] = int(top_k)
        if top_p is not None and top_p > 0:
            gen_kwargs["top_p"] = float(top_p)

        import contextlib

        autocast = (
            torch.autocast("cpu", dtype=torch.bfloat16)
            if (self.bf16 and device == "cpu")
            else contextlib.nullcontext()
        )
        with torch.no_grad(), autocast:
            audio_values = self._model.generate(**inputs, **gen_kwargs)

        sr = self.sample_rate
        out: List[GeneratedAudio] = []
        # audio_values: [batch, channels, samples]
        arr = audio_values.detach().cpu().numpy()
        for i, prompt in enumerate(prompts):
            wav = self._normalize(np.asarray(arr[i, 0], dtype=np.float32))
            out.append(
                GeneratedAudio(
                    waveform=wav,
                    sample_rate=sr,
                    duration_s=float(len(wav) / sr),
                    prompt=prompt,
                )
            )
        return out

    def generate_candidates(
        self,
        prompt: str,
        n: int = 3,
        duration_s: float = MUSICGEN_DEFAULT_DURATION_S,
        base_seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> List[GeneratedAudio]:
        """Render ``n`` candidate clips (distinct seeds) for best-of selection."""
        n = max(1, int(n))
        candidates: List[GeneratedAudio] = []
        for i in range(n):
            seed = None if base_seed is None else int(base_seed) + i
            candidates.append(
                self.generate(
                    prompt, duration_s=duration_s, seed=seed,
                    guidance_scale=guidance_scale, temperature=temperature,
                    top_k=top_k, top_p=top_p,
                )
            )
        return candidates

    # -- IO ------------------------------------------------------------------

    @staticmethod
    def write_wav(audio: GeneratedAudio, out_path: str | Path) -> str:
        """Write a clip to a WAV file and return the path."""
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("soundfile is required to write WAV files") from exc

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio.waveform, audio.sample_rate)
        return str(out.as_posix())

    def generate_to_file(
        self,
        prompt: str,
        out_path: str | Path,
        duration_s: float = MUSICGEN_DEFAULT_DURATION_S,
        seed: Optional[int] = None,
    ) -> tuple[str, GeneratedAudio]:
        """Generate and persist a clip in one call."""
        audio = self.generate(prompt, duration_s=duration_s, seed=seed)
        path = self.write_wav(audio, out_path)
        return path, audio
