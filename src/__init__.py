"""
Music Generation & Emotion Detection System
Version: 1.0
"""

__version__ = "1.0.0"
__author__ = "Solo Developer"

# Subpackages are imported explicitly where needed (e.g. `from src.config.config
# import ...`). We deliberately do NOT eagerly import them here — that would force the
# serving app (config + music_generation + api) to drag in the heavy phase-1 data
# pipeline (HDF5 dataloaders), which isn't needed and isn't shipped to the GPU/Space.

__all__ = ["__version__", "__author__"]

