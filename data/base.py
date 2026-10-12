"""Abstract contract for pluggable data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from data.dataset import Dataset


class DataSource(ABC):
    """A source that loads a dataset without depending on model code."""

    @abstractmethod
    def load(self) -> Dataset:
        """Load and return a validated dataset."""
