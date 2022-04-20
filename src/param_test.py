
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import copy
import math
import numpy as np
import datetime

from torch.utils.tensorboard import SummaryWriter


###torch.manual_seed(42)
torch.backends.cudnn.deterministic = 2
torch.backends.cudnn.benchmark = False
torch.set_printoptions(precision=10)
device = 'cpu'


max_epochs = 2000

# number of random samples to generate (should be a multiple of two for flattening an IQ pair)
io_size = 1
feature_size = 1
decoder_int1 = 1

read_data = False

class scale_net(nn.Module):
    def __init__(self, fp_max, fp_min):
        super().__init__()
        # self.scale = nn.Parameter(data=torch.tensor(1.0), requires_grad=True)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        # self.scale = nn.Linear(1, 1, bias=False)
        self.fp_max = fp_max
        self.fp_min = fp_min

    def forward(self, feature, dec):
        # d_ol = copy.deepcopy(dec.output_layer.weight)

        # d_ol_q = torch.clamp_min(torch.clamp_max(torch.floor(self.scale(d_ol)), self.fp_max), self.fp_min)
        # dec.output_layer.weight.data = torch.floor(d_ol_q + 0.5)/self.scale.weight.data

        # d_ol_q = torch.clamp_min(torch.clamp_max(torch.floor(d_ol * self.scale), self.fp_max), self.fp_min)
        # dec.output_layer.weight.data = torch.floor(d_ol_q + 0.5)/self.scale

        # xh = torch.floor(dec(feature) + 0.5)
        xh = feature * self.scale
        return xh



if __name__ == '__main__':

    idx = 0
    rng = np.random.default_rng(10)
    data_bits = 12
    data_min = 0
    data_max = 2**data_bits
    fp_range = 2**data_bits      # the max value

    # the min/max number (0 <= x < fp_range)
    fp_min = 0
    fp_max = fp_range - 1

    if(read_data == True):
        x = np.fromfile("../data/lfm_test_10M_100m_0000.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)
        x = x[np.s_[idx:(idx + io_size)]]
        data_type = "real"
    else:
        # normal range of IQ values
        #x = rng.integers(-2048, 2048, size=(1, 1, 1, io_size), dtype=np.int16, endpoint=False).astype(np.float32)
        # normal range of IQ values converted to unsigned with a shift
        #x = rng.integers(0, 4096, size=(1, 1, 1, io_size), dtype=np.int16, endpoint=False).astype(np.float32)
        # normal IQ values decomposed into 8-bit unsigned values
        x = rng.integers(data_min, data_max, size=(1, 1, 1, io_size), dtype=np.int16, endpoint=False).astype(np.float32)
        data_type = "12bit-uint"

    # get the mean of x
    x_mean = math.floor(np.mean(x))

    # input into the decoder
    F = torch.from_numpy(2048*np.ones((1, 1, 1, feature_size)).astype(np.float32)).to(device)
    F = F.view(-1, feature_size)

    # convert x into a torch tensor variable
    X = torch.from_numpy(x).to(device)
    X = X.view(-1, io_size)

    device = "cpu"
    sn = scale_net(fp_max, fp_min).to(device)

    sn.train()
    print(list(sn.parameters()))

    # scale = nn.Parameter(torch.tensor(128.0, device=device), requires_grad=True)
    scale_optimizer = optim.Adam(sn.parameters(), lr=1e-3)
    scale_crit = nn.MSELoss()

    for epoch in range(max_epochs):
        loss = 0
        scale_optimizer.zero_grad()

        # dw2b = copy.deepcopy(model.decoder.output_layer.weight)
        # dw2b_q = torch.clamp_min(torch.clamp_max(torch.floor(dw2b * scale), fp_max), fp_min)
        # d2 = torch.floor(dw2b_q + 0.5)/scale
        #
        # D = copy.deepcopy(model.decoder)
        # D.output_layer.weight.data = d2
        # Y2 = torch.floor(D(F) + 0.5)

        Y2 = sn(F, 1)

        scale_loss = scale_crit(Y2, X)

        loss = scale_loss.item()
        scale_loss.backward()
        scale_optimizer.step()
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, loss))


    bp = 1
