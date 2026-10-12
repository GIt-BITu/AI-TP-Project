"""Abstract interface shared by all network layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class Layer(ABC):
    """A component that transforms one vector into another."""

    @abstractmethod
    def forward(self, inputs: Sequence[float]) -> list[float]:
        """Compute and return this layer's output."""
