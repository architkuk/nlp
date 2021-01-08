import torch
import torch.nn as nn
import torch.nn.functional as F

class VQA(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(VQA, self).__init__()
        self.lin1 = nn.Linear()
        self.gru = nn.GRU()
    def forward(self, x):
        yTilde = F.tanh()
        g = F.sigmoid()
        y = yTilde * g
        return x