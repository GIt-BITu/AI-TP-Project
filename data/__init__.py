"""Data handling, kept independent from neural-network implementation."""

from data.dataset import Dataset
from data.xor_source import XORDataSource

__all__ = ["Dataset", "XORDataSource"]
