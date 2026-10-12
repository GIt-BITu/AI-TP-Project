"""Loss functions and output-layer gradients."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence

from neural_network.activations import Activation


class LossFunction(ABC):
    """Interface for a loss function used by the output layer."""

    name: str

    @abstractmethod
    def calculate(self, target: Sequence[float], prediction: Sequence[float]) -> float:
        """Return the scalar loss for one target/prediction pair."""

    @abstractmethod
    def gradient_wrt_logits(
        self,
        target: Sequence[float],
        prediction: Sequence[float],
        output_activation: Activation,
    ) -> list[float]:
        """Return d(loss)/d(z), where z is the output layer's pre-activation."""

    @staticmethod
    def _check_pair(target: Sequence[float], prediction: Sequence[float]) -> None:
        if len(target) == 0:
            raise ValueError("Targets and predictions must not be empty.")
        if len(target) != len(prediction):
            raise ValueError(
                "Target and prediction sizes differ: "
                f"{len(target)} versus {len(prediction)}."
            )


class MeanSquaredError(LossFunction):
    """Mean squared error, averaged over the output units of one sample."""

    name = "mse"

    def calculate(self, target: Sequence[float], prediction: Sequence[float]) -> float:
        self._check_pair(target, prediction)
        return sum((actual - predicted) ** 2 for actual, predicted in zip(target, prediction)) / len(target)

    def gradient_wrt_logits(
        self,
        target: Sequence[float],
        prediction: Sequence[float],
        output_activation: Activation,
    ) -> list[float]:
        self._check_pair(target, prediction)
        output_count = len(target)
        return [
            (2.0 * (predicted - actual) / output_count)
            * output_activation.derivative_from_output(predicted)
            for actual, predicted in zip(target, prediction)
        ]


class BinaryCrossEntropy(LossFunction):
    """Binary cross-entropy for sigmoid output units."""

    name = "binary_cross_entropy"
    _EPSILON = 1e-12

    def calculate(self, target: Sequence[float], prediction: Sequence[float]) -> float:
        self._check_pair(target, prediction)
        total = 0.0
        for index, (actual, predicted) in enumerate(zip(target, prediction)):
            if not 0.0 <= actual <= 1.0:
                raise ValueError(f"Binary cross-entropy target at index {index} must be in [0, 1].")
            probability = min(max(predicted, self._EPSILON), 1.0 - self._EPSILON)
            total -= actual * math.log(probability) + (1.0 - actual) * math.log(1.0 - probability)
        return total / len(target)

    def gradient_wrt_logits(
        self,
        target: Sequence[float],
        prediction: Sequence[float],
        output_activation: Activation,
    ) -> list[float]:
        self._check_pair(target, prediction)
        if output_activation.name != "sigmoid":
            raise ValueError("Binary cross-entropy requires a sigmoid output activation.")
        for index, actual in enumerate(target):
            if not 0.0 <= actual <= 1.0:
                raise ValueError(f"Binary cross-entropy target at index {index} must be in [0, 1].")
        # The sigmoid derivative cancels algebraically with BCE's probability derivative.
        return [(predicted - actual) / len(target) for actual, predicted in zip(target, prediction)]


_LOSSES: dict[str, LossFunction] = {
    "mse": MeanSquaredError(),
    "mean_squared_error": MeanSquaredError(),
    "binary_cross_entropy": BinaryCrossEntropy(),
    "bce": BinaryCrossEntropy(),
}


def get_loss(name: str | LossFunction) -> LossFunction:
    """Resolve a built-in loss or accept a custom LossFunction implementation."""
    if isinstance(name, LossFunction):
        return name
    if not isinstance(name, str):
        raise TypeError("Loss must be specified by name or as a LossFunction object.")
    key = name.strip().lower()
    try:
        return _LOSSES[key]
    except KeyError as exc:
        options = "mse, binary_cross_entropy"
        raise ValueError(f"Unsupported loss {name!r}. Choose from: {options}.") from exc
