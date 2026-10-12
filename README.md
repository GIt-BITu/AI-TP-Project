# Dynamic Neural Network from Scratch (Python)

A small, extensible feed-forward neural network for learning and experimentation. The forward pass, activation functions, loss gradients, backpropagation, parameter initialization, and stochastic gradient descent updates are implemented directly in Python. It does **not** use NumPy, PyTorch, TensorFlow, or another machine-learning framework.

The included runnable example trains on the XOR truth table stored in a plain-text data file.

## Requirements

- Python 3.10 or newer
- No third-party packages

## Run the example

From the project root:

```bash
python main.py
```

The entry point loads the XOR dataset, builds a `2 -> 4 -> 1` network, trains it, then prints the predicted probability and binary class for each XOR input.

Run the automated tests with:

```bash
python -m unittest discover -s tests -v
```

## Project structure

```text
dynamic_neural_network/
├── main.py                       # Orchestration / application entry point
├── data/
│   ├── base.py                   # DataSource interface
│   ├── dataset.py                # Small dataset value object
│   ├── xor_source.py             # XOR text-file reader
│   └── datasets/xor.txt          # Editable comma-separated examples
├── neural_network/
│   ├── activations.py            # Activation functions and derivatives
│   ├── losses.py                 # Loss functions and output gradients
│   ├── network.py                # Dynamic architecture and training loop
│   ├── layers/
│   │   ├── base.py               # Shared layer interface
│   │   ├── input_layer.py        # Input validation and forwarding
│   │   ├── dense.py              # Shared trainable dense-layer logic
│   │   ├── hidden_layer.py       # Hidden-layer specialization
│   │   └── output_layer.py       # Output-layer specialization
│   └── utils/validation.py       # Reusable numeric and shape validation
└── tests/                        # Standard-library unit tests
```

### Responsibilities and design

- **`main.py` orchestrates** data loading, model configuration, training, and presentation. It contains no neural-network math.
- **`data/` is independent** from model code. `DataSource` gives future data sources a common interface; the XOR source reads a small CSV-like text file. Other sources can be added without changing the network.
- **`InputLayer`** validates and forwards the input features; it has no trainable parameters.
- **`HiddenLayer`** represents each intermediate trainable layer.
- **`OutputLayer`** represents the final trainable layer and starts backpropagation from the selected loss.
- **`DenseLayer`** centralizes the matrix/vector calculations and parameter updates shared by hidden and output layers.
- **`activations.py`, `losses.py`, and `utils/validation.py`** group logic by responsibility instead of duplicating it across layers.

## Change the architecture

`hidden_sizes` is a sequence, so it can describe zero, one, or many hidden layers. For example:

```python
from neural_network import NeuralNetwork

model = NeuralNetwork(
    input_size=3,
    hidden_sizes=(8, 6, 4),  # three hidden layers
    output_size=2,
    hidden_activation="relu",
    output_activation="linear",
    loss="mse",
    seed=42,
)
```

Use `hidden_sizes=()` for no hidden layer. Supported activations are `sigmoid`, `tanh`, `relu`, and `linear`. Supported losses are `mse` and `binary_cross_entropy` (`bce` is an alias). Binary cross-entropy requires a sigmoid output activation and targets in `[0, 1]`.

## Data format

`data/datasets/xor.txt` has one sample per non-comment line:

```text
# input_1,input_2,target
0,0,0
0,1,1
1,0,1
1,1,0
```

Each row currently contains two input features and one target. `XORDataSource` skips empty lines and comments and reports malformed rows with their file path and line number.

## Training API

```python
history = model.fit(
    inputs,
    targets,
    epochs=5000,
    learning_rate=0.5,
    shuffle=True,
    target_loss=0.02,  # optional early stopping threshold
)

probabilities = model.predict([0.0, 1.0])
batch_predictions = model.predict_batch(inputs)
loss = model.evaluate_loss(inputs, targets)
```

`fit()` returns the loss after each epoch. For a single-output model, targets may be supplied either as nested rows such as `[[0], [1]]` or as a flat sequence such as `[0, 1]`.

## Implementation notes

- The network uses fully connected layers with Glorot/Xavier uniform weight initialization and zero biases.
- Training uses online stochastic gradient descent: each sample triggers a forward pass and backward update.
- The output layer calculates gradients with respect to pre-activation values, which allows binary cross-entropy and sigmoid to use their stable combined gradient.
- Input sizes, target sizes, finite numeric values, training parameters, and data shape are checked with useful errors.
- Reproducibility is supported through the `seed` constructor argument.

This is an educational, single-process implementation. It is intentionally small and does not currently include batching, optimizers beyond SGD, persistence/checkpoints, feature scaling, GPU acceleration, or a production data pipeline.
