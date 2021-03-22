
import copy

import numpy as np
import torch
import torch.nn as nn

# import parameters for model
from params import *


def init_weights(model):
    # use something like this to manually set the weights.  use the no_grad() to prevent tracking of gradient changes
    with torch.no_grad():
        for param in model.decoder.parameters():
            # rnd_range = 1/128
            mr = np.random.default_rng(10)
            param_size = param.size()
            param.data = nn.Parameter(torch.from_numpy(2 * (mr.uniform(0, 1.0, param.size()) > 0.5).astype(np.float32) - 1)).to(device)

            # make a deep copy of the weights to make sure they don't change
            ew1 = copy.deepcopy(param)

#
# # use something like this to manually set the weights.  use the no_grad() to prevent tracking of gradient changes
# with torch.no_grad():
#     #rnd_range = 1/128
#     mr = np.random.default_rng(10)
#
#     # random values of -1/1
#     model.decoder.input_layer.weight.data = nn.Parameter(torch.from_numpy(2*(mr.uniform(0, 1.0, [decoder_int1, feature_size]) > 0.5).astype(np.float32)-1)).to(device)
#     model.decoder.hidden_layer_1.weight.data = nn.Parameter(torch.from_numpy(2 * (mr.uniform(0, 1.0, [256, decoder_int1]) > 0.5).astype(np.float32) - 1)).to(device)
#     model.decoder.hidden_layer_2.weight.data = nn.Parameter(torch.from_numpy(2 * (mr.uniform(0, 1.0, [512, 256]) > 0.5).astype(np.float32) - 1)).to(device)
#     model.decoder.hidden_layer_3.weight.data = nn.Parameter(torch.from_numpy(2 * (mr.uniform(0, 1.0, [1024, 512]) > 0.5).astype(np.float32) - 1)).to(device)
#     model.decoder.hidden_layer_4.weight.data = nn.Parameter(torch.from_numpy(2 * (mr.uniform(0, 1.0, [2048, 1024]) > 0.5).astype(np.float32) - 1)).to(device)
#     model.decoder.output_layer.weight.data = nn.Parameter(torch.from_numpy(2*(mr.uniform(0, 1.0, [input_size, 2048]) > 0.5).astype(np.float32)-1)).to(device)
#
#     # make a deep copy of the weights to make sure they don't change
#     ew1 = copy.deepcopy(model.encoder.input_layer.weight)
#     dw1a = copy.deepcopy(model.decoder.input_layer.weight)
#     dw2a = copy.deepcopy(model.decoder.output_layer.weight)
#
