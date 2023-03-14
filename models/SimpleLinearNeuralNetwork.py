from torch import nn


class SimpleLinearNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x):
        logits = self.linear(x)
        return logits
