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

# from torch.utils.tensorboard import SummaryWriter

# import the network
from zsl_decoder_v1 import Decoder
from zsl_error_metric import zsl_error_metric

###torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_printoptions(precision=10)

max_epochs = 30000

# number of random samples to generate (should be a multiple of two for flattening an IQ pair)
fft_size = 2**14
io_size = 1024
feature_size = 1
decoder_int1 = 1

read_data = True

# setup everything
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model = Decoder(io_size, feature_size).to(device)

# this is setup as a static learning rate.  we may want to look at variable lr based on some performance numbers
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

def quantize_weights(W, scale):
    w_s = W * scale
    w_q = np.fmax(np.fmin(np.floor(w_s), fp_max * np.ones(W.shape)), fp_min * np.ones(W.shape))
    w_q = w_q / scale

    return w_q

# create a parameterized version of the unconstrained optimization function
def get_best_scale(scale, F, W, X, fp_min, fp_max):

    p = scale.shape[0]
    #w_s = W * scale

    f_loss = np.empty(0)

    for idx in range(0, p):
        # w_s = W * scale[idx]

        # method 1
        # w_q1 = np.fmax(np.fmin(np.floor(np.abs(w_s) + 0.5), fp_max*np.ones(W.shape)), fp_min*np.ones(W.shape))
        # w_q1 = np.copysign(w_q1, w_s)/scale[idx]

        # method 2
        # w_q = np.fmax(np.fmin(np.floor(w_s + 0.5), fp_max * np.ones(W.shape)), fp_min * np.ones(W.shape))
        # w_q = w_q / scale[idx]

        w_q = quantize_weights(W, scale[idx])

        Y = np.floor((w_q.transpose()*F).sum(axis=1) + 0.5)
        f_loss = np.append(f_loss, np.sum(np.abs(Y - X)))

    return f_loss

if __name__ == '__main__':

    idx = 0
    rng = np.random.default_rng(10)
    data_bits = 12
    data_min = 0
    data_max = 2**data_bits
    fp_bits = 4

    # input into the decoder
    F = torch.from_numpy((2048*np.ones((1, 1, 1, feature_size))).astype(np.float32)).to(device)
    F = F.view(-1, feature_size)

    print("Loading data...\n")

    base_name = "sdr_test"
    xd = np.fromfile("../data/" + base_name + "_10M_100m_0000.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)

    # base_name = "VH1-164"
    # xd = np.fromfile("e:/data/zsl/" + base_name + ".sigmf-data.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)

    x_blocks = math.ceil(xd.size/fft_size)
    data_type = "sdr"

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_name = "fs{:02d}-io{:03d}-".format(feature_size, io_size) + data_type
    log_dir = "../results/" + scenario_name + "/"

    os.makedirs(log_dir, exist_ok=True)

    # writer to save the data
    test_writer = open((log_dir + base_name + "_{:02d}-bits_M3_".format(fp_bits) + "data.bin"), "wb")

    for fft_blk in range(0, x_blocks*fft_size, fft_size):
        print("Processing FFT block {:d}".format(fft_blk))
        x = xd[fft_blk:(fft_blk + fft_size)]

        # get the mean of x
        x_mean = math.floor(np.mean(x))
        x_std = np.std(x)

        # convert x into a complex numpy array
        x = x.reshape(-1, 2)

        xc = np.empty(x.shape[0], dtype=complex)
        xc.real = x[:, 0]
        xc.imag = x[:, 1]

        # take the FFT of the block
        x_fft = np.fft.fft(xc)/xc.size

        # the FFT version that has been packed into real, imag, real, imag,...
        x_f = np.zeros([fft_size], dtype=np.float32)
        x_f[0:fft_size:2] = x_fft.real
        x_f[1:fft_size:2] = x_fft.imag
        x_f = np.floor(x_f + 0.5)

        # container for the reconstructed
        x_r = np.zeros([fft_size], dtype=np.float32)

        print("Processing...\n")
        for idx in range(0, fft_size, io_size):
            xs = x_f[idx:(idx + io_size)]

            # get the mean of x
            x_mean = math.floor(np.mean(np.abs(xs)) + 0.5)
            x_std = np.std(xs)
            x_max = np.max(np.abs(xs))

            F = torch.from_numpy((x_max*np.ones((1, 1, 1, feature_size))).astype(np.float32)).to(device)
            F = F.view(-1, feature_size)

            # convert x into a torch tensor variable
            X = torch.from_numpy(xs).to(device)
            X = X.view(-1, xs.size)

            if (xs.size < io_size):
                X = torch.nn.functional.pad(X, (0, io_size-xs.size))

            # model must be set to train mode
            model.train()

            print("block {:}".format(idx))
            for epoch in range(max_epochs):
                loss = 0
                optimizer.zero_grad()
                outputs = model(F)

                train_loss = criterion(outputs, X)
                loss += train_loss.item()

                # print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, loss))

                # loss_q = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
                loss_q = torch.sum(torch.abs(outputs - X))
                if (loss_q < 1):
                    bp = 10
                    break

                train_loss.backward()
                optimizer.step()

            loss = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
            print("loss = {:.6f}\n".format(loss.item()))

            # data_writer.write("#-------------------------------------------------------------------------------\n")
            # data_writer.write("# final_loss:\n{:.4f}\n\n".format(loss.item()))
            # data_writer.write("# X:\n")
            #
            # for idx in range(io_size):
            #     data_writer.write("{}".format((X.numpy())[0][idx]))
            #     if(idx<io_size-1):
            #         data_writer.write(", ")
            #     else:
            #         data_writer.write("\n")
            #
            # data_writer.close()

            # -------------------------------------------------------------------------------
            # training complete
            # -------------------------------------------------------------------------------

            model.eval()

            # get the weights
            dw1a = copy.deepcopy(model.output_layer.weight.data).t()

            # Set-up hyperparameters
            pso_options = {'c1': 2.1, 'c2': 2.0, 'w': 1.1}

            time.sleep(1)

            # -------------------------------------------------------------------------------
            # run PSO to find the best scale value
            # -------------------------------------------------------------------------------

            # for fp_bits in range(8, 13):

            # the min/max number (0 <= x < fp_range)
            # fp_range = 2**(fp_bits)      # the max value
            # fp_min = 0
            # fp_max = fp_range - 1

            # the min/max number (fp_range <= x < fp_range-1)
            fp_range = 2**(fp_bits-1)      # the max value
            fp_min = -fp_range
            fp_max = fp_range - 1

            scale_step = 0.005

            # min_scale = math.floor(fp_range * 0.25)
            # max_scale = (fp_range * 0.625) + scale_step
            # min_loss = 1e10
            min_scale = (2**(fp_bits-2)) *0.625
            max_scale = (fp_range * 1.1) + scale_step

            # scale_bounds = [min_scale, max_scale]
            scale_bounds = (min_scale*np.ones(1), max_scale*np.ones(1))

            # data_wr = open((log_dir + scenario_name + "_{:02d}-bits_".format(fp_bits) + date_time + ".txt"), "w")

            pso_opt = ps.single.GlobalBestPSO(n_particles=40, dimensions=1, options=pso_options, bounds=scale_bounds)

            # pso_opt.optimize(fx.sphere, iters=100)
            # cost, scale = pso_opt.optimize(get_best_scale, iters=50, F=F.numpy(), W=dw1a.numpy(), X=X.numpy(), fp_min=fp_min, fp_max=fp_max)
            scale = [8.0]

            # data_wr.write("{:0.8f}, {}, ".format(scale[0], cost))

            dw1a_q = torch.clamp_min(torch.clamp_max(torch.floor(torch.abs(dw1a * scale[0])), fp_max), fp_min)
            w_q = (torch.sign(dw1a)*dw1a_q)/scale[0]

            # w_q = quantize_weights(dw1a.numpy(), scale[0])


            # Y = torch.floor((d1.t()*F).sum(axis=1) + 0.5)
            # Y = torch.floor(torch.sum(torch.from_numpy(w_q).t()*F, dim=1) + 0.5)
            Y = torch.sum(w_q.t()*F, dim=1)
            loss2 = torch.sum(torch.abs(Y - X))/X.shape[1]

            print("scale = {:0.6f}, loss = {:.2f}".format(scale[0], loss2.item()))

            # for idx in range(io_size):
            #     data_wr.write("{}".format((Y.numpy())[idx]))
            #     if(idx<io_size-1):
            #         data_wr.write(", ")
            #     else:
            #         data_wr.write("\n")
            #
            # data_wr.close()

            # copy Y into x_r
            x_r[idx:(idx + io_size)] = np.floor(Y.numpy() + 0.5)

        # end of for idx in range(0, fft_size, math.ceil(fft_size/io_size))

        # convert the x_r back into the original samples by taking the ifft
        xr_fft = np.empty(xc.size, dtype=complex)
        xr_fft.real = x_r[0:fft_size:2]
        xr_fft.imag = x_r[1:fft_size:2]

        x_i = np.fft.ifft(xr_fft*xr_fft.size)

        y = np.zeros([fft_size], dtype=np.float32)
        y[0:fft_size:2] = np.floor(x_i.real + 0.5)
        y[1:fft_size:2] = np.floor(x_i.imag + 0.5)

        x = x.reshape(-1)
        dist_mean, dist_std, phase_mean, phase_std = zsl_error_metric(x, y)

        print("dist_mean = {:0.4f}, dist_std = {:0.4f}, phase_mean = {:0.4f}, phase_std = {:0.4f}".format(dist_mean, dist_std, phase_mean, phase_std))

        # write the reconstructed data to a binary file
        #t2 = (Y.numpy())[0:xs.size]
        #t2 = t2.reshape([xs.size]).astype(np.int16)
        test_writer.write(y.astype(np.int16))

    test_writer.close()


    # just a stopping break point before the code ends
    bp = 9
