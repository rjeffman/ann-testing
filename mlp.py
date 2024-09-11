# SPDX: gps-v3-or-later
"""Multi-Layer Perceptron implementation."""

import torch  # pylint: disable=import-error


class MultiLayerPerceptron(torch.nn.Module):
    """MLP with error backpropagation training."""

    def __init__(
        self,
        layer_configuration,
        learning_rate,
        loss_function=torch.nn.MSELoss(),
        optimizer=torch.optim.SGD,
    ):
        """Initialize model."""
        super().__init__()
        #  Define network architecture
        self.model = torch.nn.Sequential(*layer_configuration)

        # Error minimization function
        self.loss_function = loss_function

        # Optimize using Simple Gradient Descent (SGD) (Usar Adam?)
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)

    def forward(self, inputs):
        """Activate network for one input pattern."""
        return self.model(inputs)

    def __single_example(self, inputs, targets):
        """Forward and backward passes, without weight update."""
        # show example
        outputs = self.forward(inputs)
        # calculate loss
        loss = self.loss_function(outputs, targets)
        # propagate error
        loss.backward()
        # retutrn example error
        return loss.item()

    def train_epoch(self, dataset):
        """Train model which all members of the dataset."""
        progress = []
        self.optimizer.zero_grad()  # zero gradients
        for _, inputs, targets in dataset:
            train_result = self.__single_example(inputs, targets)
            self.update_weights()
            progress.append(train_result)
        return progress

    def update_weights(self):
        """Update model weights."""
        self.optimizer.step()  # update weights
        self.optimizer.zero_grad()  # zero gradients

    def evaluate_dataset(self, dataset):
        """Return the number of examples correctly classified."""
        score = 0
        for label, img, _ in dataset:
            answer = self.forward(img).detach().numpy()
            score += 1 if answer.argmax() == label else 0
        return score
