# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn
import torch.nn.functional as F
from vqa import VQA
from caption import Caption
from fasterrcnn import FasterRCNN

def train(num_epochs=0):
    x = FasterRCNN()
    y = VQA()
    z = Caption()

    for i in range(0, 10):
        break
    return num_epochs

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/