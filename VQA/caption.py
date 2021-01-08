import torch
import torch.nn as nn
import torch.nn.functional as F

class Caption(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Caption, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.lin1 = nn.Linear(input_size, hidden_size, bias=False)
        self.lin2 = nn.Linear(input_size, hidden_size, bias=False)
        self.lin3 = nn.Linear(input_size, hidden_size, bias=False)
        self.lin4 = nn.Linear(input_size, hidden_size, bias=True)
    def forward(self, x):
        output1, (h0, c0) = self.lstm1(x)
        y1 = self.lin1(x)
        y2 = self.lin2(x)
        a = self.lin3(y1 + y2)
        a = F.tanh(a)
        a = F.softmax(a)
        # vhat = sum(ai,t vi)
        output2, (h1, c0) = self.lstm2(x)
        p = F.softmax(output2)
        # product p(yt | y1:t-1)
        return