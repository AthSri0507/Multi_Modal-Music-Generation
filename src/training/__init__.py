"""Training utilities for multimodal emotion classification."""

from .losses import FocalLoss, build_loss
from .metrics import compute_classification_metrics
from .music_gan_trainer import MusicGanTrainerConfig, run_music_gan_training
from .multimodal_trainer import TrainerConfig, run_phase1_training

__all__ = [
    "FocalLoss",
    "build_loss",
    "compute_classification_metrics",
    "MusicGanTrainerConfig",
    "TrainerConfig",
    "run_music_gan_training",
    "run_phase1_training",
]
