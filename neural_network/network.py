"""Dynamic feed-forward neural network with hand-written backpropagation."""

from __future__ import annotations

import random
from numbers import Real
from typing import Sequence

from neural_network.activations import Activation, get_activation
from neural_network.layers import HiddenLayer, InputLayer, OutputLayer
from neural_network.losses import LossFunction, get_loss
from neural_network.utils.validation import finite_float, numeric_vector, positive_int


class NeuralNetwork:
    """Configurable multilayer perceptron trained with stochastic gradient descent.

    ``hidden_sizes`` can contain any number of positive integers, including an
    empty sequence for a network with no hidden layers. All numerical operations,
    activations, loss gradients, and updates are implemented using the Python
    standard library; no deep-learning framework is required.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        *,
        hidden_activation: str | Activation = "tanh",
        output_activation: str | Activation = "sigmoid",
        loss: str | LossFunction = "binary_cross_entropy",
        seed: int | None = 42,
    ) -> None:
        self.input_size = positive_int(input_size, "input_size")
        self.output_size = positive_int(output_size, "output_size")
        if isinstance(hidden_sizes, (str, bytes)):
            raise TypeError("hidden_sizes must be a sequence of positive integers.")
        try:
            self.hidden_sizes = tuple(hidden_sizes)
        except TypeError as exc:
            raise TypeError("hidden_sizes must be a sequence of positive integers.") from exc
        for index, size in enumerate(self.hidden_sizes):
            positive_int(size, f"hidden_sizes[{index}]")

        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer or None.")
        self._rng = random.Random(seed)
        self.hidden_activation = get_activation(hidden_activation)
        self.output_activation = get_activation(output_activation)
        self.loss_function = get_loss(loss)

        self.input_layer = InputLayer(self.input_size)
        self.hidden_layers: list[HiddenLayer] = []
        previous_size = self.input_size
        for size in self.hidden_sizes:
            self.hidden_layers.append(
                HiddenLayer(previous_size, size, self.hidden_activation, self._rng)
            )
            previous_size = size
        self.output_layer = OutputLayer(
            previous_size, self.output_size, self.output_activation, self._rng
        )

    def forward(self, inputs: Sequence[float]) -> list[float]:
        """Run one input vector through the complete network."""
        values = self.input_layer.forward(inputs)
        for layer in self.hidden_layers:
            values = layer.forward(values)
        return self.output_layer.forward(values)

    def predict(self, inputs: Sequence[float]) -> list[float]:
        """Return one prediction vector."""
        return self.forward(inputs)

    def predict_batch(self, inputs: Sequence[Sequence[float]]) -> list[list[float]]:
        """Predict each sample in a dataset and return one vector per sample."""
        samples = self._normalize_inputs(inputs)
        return [self.predict(sample) for sample in samples]

    def fit(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]] | Sequence[Real],
        *,
        epochs: int = 5_000,
        learning_rate: float = 0.5,
        shuffle: bool = True,
        target_loss: float | None = None,
    ) -> list[float]:
        """Train with online SGD and return the loss after each epoch.

        ``target_loss`` enables early stopping when the post-epoch loss reaches
        the requested threshold. Inputs and targets should both be batched; for
        single-output networks, flat targets such as ``[0, 1, 1, 0]`` are accepted.
        """
        epoch_count = positive_int(epochs, "epochs")
        rate = finite_float(learning_rate, "learning_rate")
        if rate <= 0.0:
            raise ValueError("learning_rate must be greater than zero.")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a bool.")
        stop_loss = None if target_loss is None else finite_float(target_loss, "target_loss")
        if stop_loss is not None and stop_loss < 0.0:
            raise ValueError("target_loss must be zero or greater.")

        x_values = self._normalize_inputs(inputs)
        y_values = self._normalize_targets(targets, len(x_values))
        sample_indices = list(range(len(x_values)))
        history: list[float] = []

        for _epoch in range(epoch_count):
            if shuffle:
                self._rng.shuffle(sample_indices)
            for index in sample_indices:
                self.forward(x_values[index])
                gradient = self.output_layer.backward_from_loss(
                    y_values[index], self.loss_function, rate
                )
                for layer in reversed(self.hidden_layers):
                    gradient = layer.backward(gradient, rate)
            epoch_loss = self.evaluate_loss(x_values, y_values)
            history.append(epoch_loss)
            if stop_loss is not None and epoch_loss <= stop_loss:
                break
        return history

    def evaluate_loss(
        self,
        inputs: Sequence[Sequence[float]],
        targets: Sequence[Sequence[float]] | Sequence[Real],
    ) -> float:
        """Return the average loss across a dataset without updating parameters."""
        x_values = self._normalize_inputs(inputs)
        y_values = self._normalize_targets(targets, len(x_values))
        total = 0.0
        for sample, target in zip(x_values, y_values):
            prediction = self.forward(sample)
            total += self.loss_function.calculate(target, prediction)
        return total / len(x_values)

    def _normalize_inputs(self, inputs: Sequence[Sequence[float]]) -> list[list[float]]:
        if isinstance(inputs, (str, bytes)):
            raise TypeError("inputs must be a collection of feature vectors.")
        try:
            if len(inputs) == 0:
                raise ValueError("Dataset must contain at least one sample.")
        except TypeError as exc:
            raise TypeError("inputs must be a collection of feature vectors.") from exc
        return [
            numeric_vector(sample, self.input_size, f"inputs[{index}]")
            for index, sample in enumerate(inputs)
        ]

    def _normalize_targets(
        self,
        targets: Sequence[Sequence[float]] | Sequence[Real],
        sample_count: int,
    ) -> list[list[float]]:
        if isinstance(targets, (str, bytes)):
            raise TypeError("targets must be a collection of target vectors.")
        try:
            if len(targets) != sample_count:
                raise ValueError(
                    f"There must be one target for each sample: expected {sample_count}, "
                    f"got {len(targets)}."
                )
            if len(targets) == 0:
                raise ValueError("Dataset targets must not be empty.")
        except TypeError as exc:
            raise TypeError("targets must be a collection of target vectors.") from exc

        first = targets[0]
        if self.output_size == 1 and isinstance(first, Real) and not isinstance(first, bool):
            return [
                numeric_vector([target], 1, f"targets[{index}]")
                for index, target in enumerate(targets)  # type: ignore[list-item]
            ]
        return [
            numeric_vector(target, self.output_size, f"targets[{index}]")
            for index, target in enumerate(targets)  # type: ignore[arg-type]
        ]
