"""EmbedSim - A Python library for computing Embedding Similarity and coherence scores."""

from importlib.metadata import version

from .embedsim import groupsim, pairsim

__version__ = version("embedsim")

__all__ = [
    "groupsim",
    "pairsim",
]
