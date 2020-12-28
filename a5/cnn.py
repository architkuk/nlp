#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, e_char, e_word):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=e_char, out_channels=e_word, kernel_size=5, padding=1)
        self.maxpool = nn.MaxPool1d(19)
    # x = x_reshaped
    def forward(self, x):
        x_conv = self.conv(x)
        x_cout = self.maxpool(F.relu(x_conv))
        return x_cout

    ### END YOUR CODE

