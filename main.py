"""Application entry point: configure data, model, training, and output here."""

from __future__ import annotations

from data import XORDataSource
from neural_network import NeuralNetwork


def main() -> None:
    """Orchestrate the XOR demo; the model and data source remain independent."""
    dataset = XORDataSource().load()

    model = NeuralNetwork(
        input_size=len(dataset.inputs[0]),
        hidden_sizes=(3,),
        output_size=len(dataset.targets[0]),
        hidden_activation="tanh",
        output_activation="sigmoid",
        loss="binary_cross_entropy",
        seed=7,
    )

    history = model.fit(
        dataset.inputs,
        dataset.targets,
        epochs=10_000,
        learning_rate=0.5,
        shuffle=True,
        target_loss=0.003,
    )

    print("Dynamic neural network — XOR demonstration")
    print(f"Architecture: {model.input_size} -> {' -> '.join(map(str, model.hidden_sizes))} -> {model.output_size}")
    print(f"Training epochs: {len(history)}")
    print(f"Final loss: {history[-1]:.6f}")
    print("\nPredictions:")
    print(" input_1  input_2  expected  predicted  class")
    for features, target, prediction in zip(
        dataset.inputs, dataset.targets, model.predict_batch(dataset.inputs)
    ):
        predicted_class = int(prediction[0] >= 0.5)
        print(
            f" {int(features[0]):>7}  {int(features[1]):>7}  "
            f"{int(target[0]):>8}  {prediction[0]:>9.4f}  {predicted_class:>5}"
        )


if __name__ == "__main__":
    main()
