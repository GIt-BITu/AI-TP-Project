"""Shared trainable dense-layer implementation used by hidden and output layers."""

from __future__ import annotations

import math
import random
from typing import Sequence

from neural_network.activations import Activation, get_activation
from neural_network.layers.base import Layer
from neural_network.utils.validation import finite_float, numeric_vector, positive_int


class DenseLayer(Layer):
    """A fully connected layer with one bias and one activation per output unit.

    Weights are stored as ``weights[output_index][input_index]``. Backpropagation
    returns the gradient with respect to the layer's input and updates parameters
    using stochastic gradient descent.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str | Activation,
        rng: random.Random,
    ) -> None:
        self.input_size = positive_int(input_size, "input_size")
        self.output_size = positive_int(output_size, "output_size")
        self.activation = get_activation(activation)
        if not isinstance(rng, random.Random):
            raise TypeError("rng must be an instance of random.Random.")

        # Glorot/Xavier uniform initialization helps keep initial signals in range.
        limit = math.sqrt(6.0 / (self.input_size + self.output_size))
        self.weights = [
            [rng.uniform(-limit, limit) for _ in range(self.input_size)]
            for _ in range(self.output_size)
        ]
        self.biases = [0.0 for _ in range(self.output_size)]
        self._last_input: list[float] | None = None
        self._last_output: list[float] | None = None

    def forward(self, inputs: Sequence[float]) -> list[float]:
        values = numeric_vector(inputs, self.input_size, "layer inputs")
        outputs: list[float] = []
        for row, bias in zip(self.weights, self.biases):
            weighted_sum = sum(weight * value for weight, value in zip(row, values)) + bias
            outputs.append(self.activation(weighted_sum))
        self._last_input = values
        self._last_output = outputs
        return outputs

    def backward(self, gradient_wrt_output: Sequence[float], learning_rate: float) -> list[float]:
        """Backpropagate d(loss)/d(output), applying this layer's activation derivative."""
        if self._last_output is None:
            raise RuntimeError("forward() must be called before backward().")
        gradient = numeric_vector(gradient_wrt_output, self.output_size, "gradient_wrt_output")
        delta = [
            upstream * self.activation.derivative_from_output(output)
            for upstream, output in zip(gradient, self._last_output)
        ]
        return self._backward_from_delta(delta, learning_rate)

    def _backward_from_delta(self, delta: Sequence[float], learning_rate: float) -> list[float]:
        """Backpropagate d(loss)/d(pre-activation) and update parameters."""
        if self._last_input is None:
            raise RuntimeError("forward() must be called before backward().")
        rate = finite_float(learning_rate, "learning_rate")
        if rate <= 0.0:
            raise ValueError("learning_rate must be greater than zero.")
        deltas = numeric_vector(delta, self.output_size, "delta")

        # Calculate input gradients BEFORE changing weights; these gradients must
        # correspond to the same parameter values used in the forward pass.
        gradient_wrt_input = [
            sum(self.weights[out_index][in_index] * deltas[out_index]
                for out_index in range(self.output_size))
            for in_index in range(self.input_size)
        ]

        for output_index, unit_delta in enumerate(deltas):
            for input_index, input_value in enumerate(self._last_input):
                self.weights[output_index][input_index] -= rate * unit_delta * input_value
            self.biases[output_index] -= rate * unit_delta

        return gradient_wrt_input
