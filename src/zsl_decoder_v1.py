'''
This file is the definition of the network
'''

import os
import time

import torch.nn as nn



# create the decoder class
class Decoder(nn.Module):
    def __init__(self, output_size, feature_size):
        super().__init__()
        self.output_layer = nn.Linear(feature_size, output_size, bias=False)

    def forward(self, activation):
        activation = self.output_layer(activation)
        return activation

