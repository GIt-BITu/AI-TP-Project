"""Activation functions and their derivatives, implemented without ML libraries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Activation:
    """An activation and its derivative expressed in terms of its output."""

    name: str
    function: Callable[[float], float]
    derivative_from_output: Callable[[float], float]

    def __call__(self, value: float) -> float:
        return self.function(value)


def _sigmoid(value: float) -> float:
    # This form avoids overflow for large-magnitude negative inputs.
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _relu(value: float) -> float:
    return max(0.0, value)


_ACTIVATIONS: dict[str, Activation] = {
    "sigmoid": Activation("sigmoid", _sigmoid, lambda output: output * (1.0 - output)),
    "tanh": Activation("tanh", math.tanh, lambda output: 1.0 - output * output),
    "relu": Activation("relu", _relu, lambda output: 1.0 if output > 0.0 else 0.0),
    "linear": Activation("linear", lambda value: value, lambda _output: 1.0),
}


def get_activation(name: str | Activation) -> Activation:
    """Resolve a supported activation by name or return an Activation object."""
    if isinstance(name, Activation):
        return name
    if not isinstance(name, str):
        raise TypeError("Activation must be specified by name or as an Activation object.")
    key = name.strip().lower()
    try:
        return _ACTIVATIONS[key]
    except KeyError as exc:
        options = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(f"Unsupported activation {name!r}. Choose from: {options}.") from exc
