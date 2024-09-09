# SPDX: gps-v3-or-later
"""Implement ANN training/testing."""

from random import randint

import torch  # pylint: disable=import-error

from mlp import MultiLayerPerceptron
from mnistdata import MNISTDataset


def random_classification_test(classifier, dataset, count):
    """Check classification of 'count' random items from the dataset."""
    for index in (randint(0, len(dataset)) for _ in range(count)):
        expected, img, _ = dataset[index]
        output = classifier.forward(img).detach().numpy()
        answer = output.argmax()
        print(
            f"{index:<04d}",
            answer,
            expected,
            f"{100*output[answer]:0.3f}",
            "=" if answer == expected else [f"{o:0.1f}" for o in output],
        )


def create_network():
    """Initialize ANN model."""
    # Network architecture
    input_size = 784
    hidden_layer_size = [150]
    output_size = 10
    layers = [
        *(  # input to hidden weights
            torch.nn.Linear(input_size, hidden_layer_size[0]),
            torch.nn.LeakyReLU(),  # layer activation function
            torch.nn.LayerNorm(hidden_layer_size[0]),  # normalize weights
        ),
        *[
            cfg
            for layer in [
                (
                    torch.nn.Linear(
                        hidden_layer_size[i],
                        hidden_layer_size[i+1]
                    ),
                    torch.nn.LeakyReLU(),
                    torch.nn.LayerNorm(hidden_layer_size[i]),
                )
                for i in range(len(hidden_layer_size)-1)
            ] for cfg in layer
        ],
        *(  # hidden to output weights
            torch.nn.Linear(hidden_layer_size[-1], output_size),
            torch.nn.LeakyReLU()  # output activation function
        ),
    ]

    return MultiLayerPerceptron(
        layers,
        learning_rate=0.01,
    )


def __mlp_main():
    classifier = create_network()
    train_dataset = MNISTDataset(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte"
    )
    test_dataset = MNISTDataset(
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte"
    )

    epochs = 1
    progress = []

    # Create plot
    # fig, ax = plot.subplots()

    for _ in range(epochs):  # Train in epochs.

        progress.extend(classifier.train_epoch(train_dataset))

        # Test current model
        train_score = classifier.evaluate_dataset(train_dataset)
        test_score = classifier.evaluate_dataset(test_dataset)

        # Plot train loss
        # plot.title(f"Train Loss")
        # ax.scatter(range(len(progress)), progress, s=15)
        # plot.show()

        # print current model performance.
        perc = 100 * train_score / len(train_dataset)
        print(f"Train: {train_score} {perc:0.2f}%")
        perc = 100 * test_score / len(test_dataset)
        print(f"Test: {test_score:} {perc:0.2f}%")

    random_classification_test(classifier, test_dataset, 20)


if __name__ == "__main__":
    __mlp_main()
