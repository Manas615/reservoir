import torch
from torch import nn


class FlowLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True, dropout=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]
