"""Prompt-to-control mapping using BERT embeddings plus lightweight priors."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.config.config import BERT_MODEL_NAME, TEXT_MAX_LENGTH
from src.music_generation.control_schema import MusicControlSchema, MusicControlSpec


@dataclass
class PromptControlResult:
    """Resolved control specification and debug metadata for a text prompt."""

    spec: MusicControlSpec
    prompt: str
    emotion_scores: Dict[str, float]
    style_scores: Dict[str, float]
    instrumentation_scores: Dict[str, float]


class PromptToControlMapper:
    """Map free-form text prompts into control vectors expected by GAN generation."""

    def __init__(
        self,
        control_schema: MusicControlSchema,
        model_name: str = BERT_MODEL_NAME,
        max_length: int = TEXT_MAX_LENGTH,
        device: str = "auto",
    ) -> None:
        self.control_schema = control_schema
        self.max_length = int(max_length)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self._emotion_labels = list(self.control_schema.emotion_to_id.keys())
        self._style_labels = list(self.control_schema.style_labels)
        self._instr_labels = list(self.control_schema.instrumentation_labels)

        self._emotion_emb = self._embed_labels(self._emotion_labels, mode="emotion")
        self._style_emb = self._embed_labels(self._style_labels, mode="style")
        self._instr_emb = self._embed_labels(self._instr_labels, mode="instrumentation")

    def _embed_text(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            encoded = self.tokenizer(
                [text],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            out = self.model(**encoded)
            cls = out.last_hidden_state[:, 0, :]
            return F.normalize(cls, p=2, dim=1)

    def _prototype_phrases(self, label: str, mode: str) -> list[str]:
        key = label.strip().lower()
        if mode == "emotion":
            if key in {"q1", "happy", "joy", "joyful", "energetic"}:
                return [
                    "happy energetic uplifting music",
                    "bright joyful fast tempo",
                    "positive high energy soundtrack",
                ]
            if key in {"q2", "angry", "tense", "aggressive"}:
                return [
                    "tense dramatic aggressive music",
                    "high arousal negative emotion soundtrack",
                    "intense suspense action cue",
                ]
            if key in {"q3", "sad", "melancholy", "dark"}:
                return [
                    "sad melancholic emotional music",
                    "dark slow introspective soundtrack",
                    "low valence low energy mood",
                ]
            if key in {"q4", "calm", "peaceful", "soft"}:
                return [
                    "calm peaceful gentle music",
                    "soft relaxed low arousal positive mood",
                    "serene piano background",
                ]
            return [f"{key} emotional music", key]

        if mode == "style":
            table = {
                "generic": ["general instrumental", "balanced modern style"],
                "jazz": ["jazz swing groove", "blue notes syncopated"],
                "ambient": ["ambient atmospheric texture", "slow evolving pads"],
                "cinematic": ["cinematic soundtrack orchestral drama", "film score"],
                "piano": ["solo piano expressive", "acoustic piano piece"],
                "electronic": ["electronic synth driven", "edm style"],
            }
            return table.get(key, [key])

        if mode == "instrumentation":
            table = {
                "ensemble": ["full ensemble instruments", "layered arrangement"],
                "jazz_combo": ["jazz combo piano bass drums", "small jazz band"],
                "solo_piano": ["solo piano only", "single piano instrument"],
                "strings": ["string section orchestral", "violin cello ensemble"],
                "synth_pad": ["synth pad electronic texture", "ambient synth layers"],
            }
            return table.get(key, [key])

        return [key]

    def _embed_labels(self, labels: Sequence[str], mode: str) -> torch.Tensor:
        vectors = []
        for label in labels:
            phrases = self._prototype_phrases(label, mode=mode)
            emb = [self._embed_text(p) for p in phrases]
            stacked = torch.cat(emb, dim=0)
            mean = F.normalize(stacked.mean(dim=0, keepdim=True), p=2, dim=1)
            vectors.append(mean)
        return torch.cat(vectors, dim=0)

    def _scores(self, prompt_emb: torch.Tensor, label_emb: torch.Tensor, labels: Sequence[str]) -> Dict[str, float]:
        sims = torch.matmul(prompt_emb, label_emb.T).detach().cpu().numpy()[0]
        return {label: float(score) for label, score in zip(labels, sims)}

    @staticmethod
    def _keyword_bonus(prompt: str, mode: str) -> Dict[str, float]:
        p = prompt.lower()
        if mode == "emotion":
            bonus = {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0}
            if any(w in p for w in ["happy", "joy", "uplifting", "bright", "positive"]):
                bonus["q1"] += 0.35
                bonus["q4"] += 0.10
            if any(w in p for w in ["energetic", "upbeat", "fast", "dance", "driving"]):
                bonus["q1"] += 0.25
                bonus["q4"] += 0.10
            if any(w in p for w in ["calm", "peaceful", "soft", "relax", "chill"]):
                bonus["q4"] += 0.45
            if any(w in p for w in ["sad", "melancholy", "dark", "lonely", "heartbreak"]):
                bonus["q3"] += 0.55
            if any(w in p for w in ["tense", "angry", "aggressive", "epic", "intense"]):
                bonus["q2"] += 0.50
            return bonus

        if mode == "style":
            bonus = {
                "generic": 0.0,
                "jazz": 0.0,
                "ambient": 0.0,
                "cinematic": 0.0,
                "piano": 0.0,
                "electronic": 0.0,
            }
            if "jazz" in p or "swing" in p:
                bonus["jazz"] += 0.60
            if any(w in p for w in ["ambient", "atmospheric", "pad", "drone"]):
                bonus["ambient"] += 0.55
            if any(w in p for w in ["cinematic", "orchestral", "film", "trailer"]):
                bonus["cinematic"] += 0.55
            if "piano" in p:
                bonus["piano"] += 0.65
            if any(w in p for w in ["electronic", "synth", "edm", "techno"]):
                bonus["electronic"] += 0.60
            return bonus

        if mode == "instrumentation":
            bonus = {
                "ensemble": 0.0,
                "jazz_combo": 0.0,
                "solo_piano": 0.0,
                "strings": 0.0,
                "synth_pad": 0.0,
            }
            if any(w in p for w in ["jazz combo", "double bass", "brush drums"]):
                bonus["jazz_combo"] += 0.70
            if any(w in p for w in ["solo piano", "piano solo", "only piano"]):
                bonus["solo_piano"] += 0.80
            if any(w in p for w in ["strings", "violin", "cello", "orchestra"]):
                bonus["strings"] += 0.65
            if any(w in p for w in ["synth", "pad", "electronic"]):
                bonus["synth_pad"] += 0.60
            return bonus

        return {}

    @staticmethod
    def _extract_bpm(prompt: str) -> float | None:
        match = re.search(r"(?P<bpm>\d{2,3})\s*bpm", prompt.lower())
        if not match:
            return None
        bpm = float(match.group("bpm"))
        return float(np.clip(bpm, 40.0, 180.0))

    @staticmethod
    def _keyword_tempo(prompt: str) -> float | None:
        p = prompt.lower()
        if any(word in p for word in ["very slow", "largo", "slow", "calm", "chill"]):
            return 74.0
        if any(word in p for word in ["fast", "energetic", "upbeat", "driving", "quick"]):
            return 138.0
        if any(word in p for word in ["moderate", "medium tempo", "steady"]):
            return 112.0
        return None

    @staticmethod
    def _keyword_intensity(prompt: str) -> float | None:
        p = prompt.lower()
        if any(word in p for word in ["very soft", "gentle", "quiet", "subtle"]):
            return 0.25
        if any(word in p for word in ["soft", "calm", "mellow"]):
            return 0.4
        if any(word in p for word in ["intense", "powerful", "heavy", "aggressive", "epic"]):
            return 0.85
        if any(word in p for word in ["energetic", "strong", "punchy"]):
            return 0.72
        return None

    def map_prompt(self, prompt: str) -> PromptControlResult:
        prompt_emb = self._embed_text(prompt)

        emotion_scores = self._scores(prompt_emb, self._emotion_emb, self._emotion_labels)
        style_scores = self._scores(prompt_emb, self._style_emb, self._style_labels)
        instr_scores = self._scores(prompt_emb, self._instr_emb, self._instr_labels)

        e_bonus = self._keyword_bonus(prompt, mode="emotion")
        s_bonus = self._keyword_bonus(prompt, mode="style")
        i_bonus = self._keyword_bonus(prompt, mode="instrumentation")
        emotion_scores = {k: emotion_scores.get(k, -1e9) + e_bonus.get(k, 0.0) for k in self._emotion_labels}
        style_scores = {k: style_scores.get(k, -1e9) + s_bonus.get(k, 0.0) for k in self._style_labels}
        instr_scores = {k: instr_scores.get(k, -1e9) + i_bonus.get(k, 0.0) for k in self._instr_labels}

        emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        style = max(style_scores.items(), key=lambda x: x[1])[0]
        instrumentation = max(instr_scores.items(), key=lambda x: x[1])[0]

        default_spec = self.control_schema.default_spec(emotion)
        tempo = self._extract_bpm(prompt)
        if tempo is None:
            tempo = self._keyword_tempo(prompt)
        if tempo is None:
            tempo = float(default_spec.tempo_bpm)

        intensity = self._keyword_intensity(prompt)
        if intensity is None:
            # Map relative confidence to a bounded intensity around the default.
            top_score = max(emotion_scores.values())
            scaled = 0.5 + 0.35 * float(np.tanh(top_score))
            intensity = float(np.clip((scaled + default_spec.intensity) / 2.0, 0.0, 1.0))

        spec = MusicControlSpec(
            emotion=emotion,
            style=style,
            tempo_bpm=float(np.clip(tempo, self.control_schema.tempo_min, self.control_schema.tempo_max)),
            intensity=float(np.clip(intensity, 0.0, 1.0)),
            instrumentation=instrumentation,
        )
        return PromptControlResult(
            spec=spec,
            prompt=prompt,
            emotion_scores=emotion_scores,
            style_scores=style_scores,
            instrumentation_scores=instr_scores,
        )
