import torch
from torch import nn


class ComplexLinearNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.Tanh(),
            nn.Linear(150, 150),
            nn.ELU(),
            nn.Linear(150, 10),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)

        return logits
