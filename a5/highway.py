#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    # e_word = in/out dimensions
    def __init__(self, e_word):
        super(Highway, self).__init__()
        self.proj = nn.Linear(e_word, e_word)
        self.gate = nn.Linear(e_word, e_word)
        self.dropout = nn.Dropout(0.3)

    # x = x_conv_out
    def forward(self, x):
        x_proj = F.relu(self.proj(x))
        x_gate = torch.sigmoid(self.gate(x))
        x_hwy = x_gate * x_proj + (1 - x_gate) * x
        x_wemb = self.dropout(x_hwy)
        return x_wemb
    ### END YOUR CODE

