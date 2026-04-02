"""Music generation package for Milestone 4 pre-training foundation."""

from .dataset import EmotionConditionedMusicDataset, MusicLabelEncoder
from .losses import (
    discriminator_hinge_loss,
    generator_hinge_loss,
    r1_regularization,
)
from .midi_representation import MidiRepresentationConfig, MidiSequenceConverter
from .models import MusicDiscriminator, MusicGenerator, MusicGanOutput, build_generator
from .postprocess import MidiPostprocessConfig, MidiPostprocessor

__all__ = [
    "EmotionConditionedMusicDataset",
    "MusicDiscriminator",
    "MusicGanOutput",
    "MusicGenerator",
    "MidiRepresentationConfig",
    "MidiSequenceConverter",
    "MidiPostprocessConfig",
    "MidiPostprocessor",
    "MusicLabelEncoder",
    "build_generator",
    "discriminator_hinge_loss",
    "generator_hinge_loss",
    "r1_regularization",
]
