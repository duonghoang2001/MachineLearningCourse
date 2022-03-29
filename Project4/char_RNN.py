# Project 4: Text Generation with RNNs
# Author: Duong Hoang
# CS 460G - 001
# Due Date: Apr 11th, 2022

'''
    Purpose: predict the handwritten number using the pixel of a 28x28 image
    Pre-cond: a mnist training data csv file and a mnist testing data csv file
    Post-cond: multilayer perceptron model, predicted handwritten number, 
                and accuracy of multiclass classifier

'''

### Implementation ###

# initialize
DATA_FILE = 'tiny_shakespeare.txt'
ALPHA = 0.1                     # learning rate
NUM_OUTPUT_NODES = 5            # multiclass
NUM_HIDDEN_LAYER_NODES = 10

# import libraries
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class RNN():
    def __init__(self, data):
        self.data = data


def main():

main()