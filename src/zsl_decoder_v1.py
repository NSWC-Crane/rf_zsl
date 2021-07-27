'''
This file is the definition of the network
'''

import os
import time

import torch.nn as nn



# create the decoder class
class Decoder(nn.Module):
    def __init__(self, feature_size, output_size):
        super().__init__()
        self.output_layer = nn.Linear(feature_size, output_size, bias=False)

    def forward(self, activation):
        activation = self.output_layer(activation)
        return activation

# create the encoder class
class Encoder(nn.Module):
    def __init__(self, input_size, feature_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, feature_size, bias=False)

    def forward(self, activation):
        activation = self.input_layer(activation)
        return activation

# use the encoder and decoder classes to build the autoencoder
class AE(nn.Module):
    def __init__(self, input_size, feature_size):
        super().__init__()
        self.encoder = Encoder(input_size, feature_size)
        self.decoder = Decoder(feature_size, input_size)

    def forward(self, features):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed
