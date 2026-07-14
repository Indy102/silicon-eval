"""Tiny aggregate statistics over repeated measurements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Stat:
    """Mean/min/max summary of a set of samples."""

    mean: float
    min: float
    max: float

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> Stat:
        if not samples:
            raise ValueError("no samples to aggregate")
        return cls(mean=sum(samples) / len(samples), min=min(samples), max=max(samples))
