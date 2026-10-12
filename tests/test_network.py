from __future__ import annotations

import unittest

from data import XORDataSource
from neural_network import NeuralNetwork
from neural_network.activations import get_activation
from neural_network.losses import BinaryCrossEntropy


class NeuralNetworkTests(unittest.TestCase):
    def test_sigmoid_is_stable_for_large_inputs(self) -> None:
        sigmoid = get_activation("sigmoid")
        self.assertAlmostEqual(sigmoid(1000.0), 1.0)
        self.assertAlmostEqual(sigmoid(-1000.0), 0.0)

    def test_dynamic_architecture_can_have_multiple_hidden_layers(self) -> None:
        model = NeuralNetwork(2, (5, 3, 2), 1, seed=3)
        self.assertEqual([layer.output_size for layer in model.hidden_layers], [5, 3, 2])
        self.assertEqual(len(model.predict([0.0, 1.0])), 1)

    def test_network_can_have_no_hidden_layers(self) -> None:
        model = NeuralNetwork(2, (), 1, seed=3)
        self.assertEqual(model.hidden_layers, [])
        self.assertEqual(len(model.predict([0.0, 1.0])), 1)

    def test_forward_rejects_wrong_input_width(self) -> None:
        model = NeuralNetwork(2, (3,), 1)
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            model.predict([1.0])

    def test_binary_cross_entropy_requires_sigmoid_output(self) -> None:
        model = NeuralNetwork(2, (3,), 1, output_activation="linear", loss="bce")
        with self.assertRaisesRegex(ValueError, "requires a sigmoid"):
            model.fit([[0.0, 0.0]], [[0.0]], epochs=1)

    def test_training_learns_xor(self) -> None:
        dataset = XORDataSource().load()
        model = NeuralNetwork(
            input_size=2,
            hidden_sizes=(4,),
            output_size=1,
            hidden_activation="tanh",
            output_activation="sigmoid",
            loss=BinaryCrossEntropy(),
            seed=7,
        )
        history = model.fit(
            dataset.inputs,
            dataset.targets,
            epochs=10_000,
            learning_rate=0.5,
            shuffle=True,
            target_loss=0.02,
        )
        predictions = model.predict_batch(dataset.inputs)
        classes = [int(row[0] >= 0.5) for row in predictions]
        expected = [int(row[0]) for row in dataset.targets]
        self.assertEqual(classes, expected)
        self.assertLess(history[-1], 0.1)
        self.assertLess(history[-1], history[0])

    def test_flat_targets_are_supported_for_single_output(self) -> None:
        model = NeuralNetwork(2, (3,), 1, seed=2)
        history = model.fit(
            [[0.0, 0.0], [0.0, 1.0]], [0.0, 1.0], epochs=2, shuffle=False
        )
        self.assertEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
