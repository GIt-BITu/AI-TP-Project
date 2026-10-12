"""A minimal dataset value object, independent from model implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    """Feature rows and target rows with optional human-readable column names."""

    inputs: list[list[float]]
    targets: list[list[float]]
    input_names: tuple[str, ...] = ()
    target_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("Dataset inputs must contain at least one sample.")
        if len(self.inputs) != len(self.targets):
            raise ValueError("Dataset inputs and targets must have the same sample count.")
        if any(not row for row in self.inputs):
            raise ValueError("Feature rows must not be empty.")
        if any(not row for row in self.targets):
            raise ValueError("Target rows must not be empty.")
        input_width = len(self.inputs[0])
        target_width = len(self.targets[0])
        if any(len(row) != input_width for row in self.inputs):
            raise ValueError("All feature rows must have the same width.")
        if any(len(row) != target_width for row in self.targets):
            raise ValueError("All target rows must have the same width.")
        if self.input_names and len(self.input_names) != input_width:
            raise ValueError("input_names count must match the feature width.")
        if self.target_names and len(self.target_names) != target_width:
            raise ValueError("target_names count must match the target width.")
