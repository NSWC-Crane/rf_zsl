'''
This file is an experiment to test compression using a decoder only with fixed point and or integer based weights
'''

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import copy
import math
import numpy as np
import datetime

import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

from torch.utils.tensorboard import SummaryWriter


###torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_printoptions(precision=10)

max_epochs = 20000

# number of random samples to generate (should be a multiple of two for flattening an IQ pair)
io_size = 16
feature_size = 1
decoder_int1 = 1

read_data = False

class scale_net(nn.Module):
    def __init__(self, fp_max, fp_min):
        super().__init__()
        # self.scale = nn.Parameter(data=torch.tensor(1.0), requires_grad=True)
        self.scale = torch.nn.Parameter(torch.tensor(128.0))

    # self.scale = nn.Linear(1, 1, bias=False)
        self.fp_max = fp_max
        self.fp_min = fp_min

    def forward(self, feature, w):
        # d_ol = copy.deepcopy(dec.output_layer.weight)

        # d_ol_q = torch.clamp_min(torch.clamp_max(torch.floor(self.scale(d_ol)), self.fp_max), self.fp_min)
        # dec.output_layer.weight.data = torch.floor(d_ol_q + 0.5)/self.scale.weight.data

        w_q = torch.clamp_min(torch.clamp_max(torch.floor(w * self.scale), self.fp_max), self.fp_min)
        w_q = torch.floor(w_q + 0.5)/self.scale

        xh = torch.floor((w_q * feature) + 0.5)
        # self.mark_non_differentiable(xh)
        return xh


# create the decoder class
class Decoder(nn.Module):
    def __init__(self, output_size, feature_size):
        super().__init__()
        self.output_layer = nn.Linear(feature_size, output_size, bias=False)
        #self.alpha = nn.Parameter(torch.tensor(10.0))

    def forward(self, activation):
        activation = self.output_layer(activation)
        return activation

# setup everything
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model = Decoder(io_size, feature_size).to(device)


# this is setup as a static learning rate.  we may want to look at variable lr based on some performance numbers
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

criterion = nn.MSELoss()
#criterion = nn.L1Loss()


#cv2.setNumThreads(0)

#torch.multiprocessing.set_start_method('spawn')
if __name__ == '__main__':

    idx = 0
    rng = np.random.default_rng(10)
    data_bits = 12
    data_min = 0
    data_max = 2**data_bits

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

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_name = "fs{:02d}-io{:03d}-".format(feature_size, io_size) + data_type
    log_dir = "../results/" + scenario_name + "/"

    os.makedirs(log_dir, exist_ok=True)

    data_writer = open((log_dir + scenario_name + "_" + date_time + ".txt"), "w")

    data_writer.write("#-------------------------------------------------------------------------------\n")
    data_writer.write("# data bits, min, max:\n{}, {}, {}\n\n".format(data_bits, data_min, data_max))
    data_writer.write("# io_size:\n{}\n\n".format(io_size))
    data_writer.write("# feature_size:\n{}\n\n".format(feature_size))
    data_writer.write("# F:\n")

    for idx in range(feature_size):
        data_writer.write("{:.6f}".format((F.numpy())[0][idx]))
        if(idx < feature_size-1):
            data_writer.write(", ")
        else:
            data_writer.write("\n\n")

    # model must be set to train mode for QAT logic to work
    model.train()

    for epoch in range(max_epochs):
        loss = 0
        optimizer.zero_grad()
        outputs = model(F)

        train_loss = criterion(outputs, X)
        loss += train_loss.item()

        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, loss))

        loss_q = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
        if (loss_q < 1):
            bp = 10
            break

        train_loss.backward()
        optimizer.step()

    loss = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
    print("\nloss = {:.6f}".format(loss.item()))

    data_writer.write("#-------------------------------------------------------------------------------\n")
    data_writer.write("# final_loss:\n{:.4f}\n\n".format(loss.item()))
    data_writer.write("# X:\n")

    for idx in range(io_size):
        data_writer.write("{}".format((X.numpy())[0][idx]))
        if(idx<io_size-1):
            data_writer.write(", ")
        else:
            data_writer.write("\n")

    data_writer.close()

    # model.eval()

    # get the weights
    dw1a = copy.deepcopy(model.output_layer.weight.data).t()

    # Set-up hyperparameters
    pso_options = {'c1': 2.1, 'c2': 2.0, 'w': 1.1}

    # create a parameterized version of the classic Rosenbrock unconstrained optimzation function
    def get_best_scale(scale, F, W, X, fp_min, fp_max):

        p = scale.shape[0]
        w_s = W * scale

        f_loss = np.empty(0)

        for idx in range(0, p):
            w_q = np.fmax(np.fmin(np.floor(w_s[idx, :]), fp_max*np.ones(W.size)), fp_min*np.ones(W.size))
            w_q = np.floor(w_q + 0.5)/scale[idx]

            Y = np.floor((w_q * F) + 0.5)
            f_loss = np.append(f_loss, np.sum(np.abs(Y - X)))

        return f_loss


    for fp_bits in range(8, 13):

        fp_range = 2**fp_bits      # the max value

        # the min/max number (0 <= x < fp_range)
        fp_min = 0
        fp_max = fp_range - 1

        scale_step = 0.005

        min_scale = math.floor(fp_range * 0.25)
        max_scale = (fp_range * 0.625) + scale_step
        min_loss = 1e10

        # scale_bounds = [min_scale, max_scale]
        scale_bounds = (min_scale*np.ones(1), max_scale*np.ones(1))

        data_wr = open((log_dir + scenario_name + "_{:02d}-bits_".format(fp_bits) + date_time + ".txt"), "w")

        pso_opt = ps.single.GlobalBestPSO(n_particles=50, dimensions=1, options=pso_options, bounds=scale_bounds)

        # pso_opt.optimize(fx.sphere, iters=100)
        cost, scale = pso_opt.optimize(get_best_scale, iters=200, F=F.numpy(), W=dw1a.numpy(), X=X.numpy(), fp_min=fp_min, fp_max=fp_max)

        time.sleep(1)
        print("scale = {:0.6f}, loss = {:.2f}\n".format(scale[0], cost))

        data_wr.write("{:0.8f}, {}, ".format(scale[0], cost))

        dw1a_q = torch.clamp_min(torch.clamp_max(torch.floor(dw1a * scale[0]), fp_max), fp_min)
        d1 = torch.floor(dw1a_q + 0.5)/scale[0]

        Y = torch.floor(F*d1 + 0.5)
        # loss2 = torch.sum(torch.abs(Y - X))

        # print("loss2 = {:.2f}".format(loss2.item()))

        for idx in range(io_size):
            data_wr.write("{}".format((Y.numpy())[0][idx]))
            if(idx<io_size-1):
                data_wr.write(", ")
            else:
                data_wr.write("\n")

        data_wr.close()


    # just a stopping break point before the code ends
    bp = 9
