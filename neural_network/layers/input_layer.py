"""Input layer: validates feature vectors but has no trainable parameters."""

from __future__ import annotations

from typing import Sequence

from neural_network.layers.base import Layer
from neural_network.utils.validation import numeric_vector, positive_int


class InputLayer(Layer):
    """The network's entry point; forwards a validated input vector unchanged."""

    def __init__(self, input_size: int) -> None:
        self.input_size = positive_int(input_size, "input_size")

    def forward(self, inputs: Sequence[float]) -> list[float]:
        return numeric_vector(inputs, self.input_size, "inputs")
