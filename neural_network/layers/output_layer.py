"""Output layer specialization with loss-aware backpropagation."""

from __future__ import annotations
from typing import Sequence

from neural_network.layers.dense import DenseLayer
from neural_network.losses import LossFunction


class OutputLayer(DenseLayer):
    """The final trainable layer, responsible for initializing loss gradients."""

    def backward_from_loss(
        self,
        target: Sequence[float],
        loss: LossFunction,
        learning_rate: float,
    ) -> list[float]:
        """Compute the output-logit gradient from the configured loss and update."""
        if self._last_output is None:
            raise RuntimeError("forward() must be called before backward_from_loss().")
        delta = loss.gradient_wrt_logits(target, self._last_output, self.activation)
        return self._backward_from_delta(delta, learning_rate)
