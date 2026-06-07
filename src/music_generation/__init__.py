"""Music generation package.

Re-exports are **lazy** (PEP 562): importing this package must not eagerly pull the
v1 GAN / MIDI training modules (torch, pretty_midi, ...), so the serving app
(text-to-audio) stays light and importable with only `requirements-serve.txt`. The
public names below still resolve on first access for code that wants them.
"""

import importlib

# public name -> submodule that defines it
_LAZY = {
    "EmotionConditionedMusicDataset": "dataset",
    "MusicLabelEncoder": "dataset",
    "discriminator_hinge_loss": "losses",
    "generator_hinge_loss": "losses",
    "r1_regularization": "losses",
    "MidiRepresentationConfig": "midi_representation",
    "MidiSequenceConverter": "midi_representation",
    "MusicDiscriminator": "models",
    "MusicGenerator": "models",
    "MusicGanOutput": "models",
    "build_generator": "models",
    "MidiPostprocessConfig": "postprocess",
    "MidiPostprocessor": "postprocess",
}

__all__ = list(_LAZY.keys())


def __getattr__(name: str):  # PEP 562
    if name in _LAZY:
        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
