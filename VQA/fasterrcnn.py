import torch
import torch.nn as nn
import torch.nn.functional as F

class FasterRCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        self.lin1 = nn.Linear()
        self.lin2 = nn.Linear()
        self.lstm1 = nn.LSTM()
        self.lstm2 = nn.LSTM()
        self.meanPool = nn.AvgPool1d()
    def forward(self, x):
        x = self.lin1()
        x = self.lin2()
        x = self.lstm1()
        x = self.lstm2()
        x = self.meanPool()
        return x