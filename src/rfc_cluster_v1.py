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

from sklearn.cluster import KMeans

import pyswarms as ps

# from torch.utils.tensorboard import SummaryWriter

# import matplotlib to plot things
from matplotlib import pyplot as plt

# scipy curve fitting
from scipy.optimize import curve_fit, fmin_bfgs

# import the network
from zsl_decoder_v1 import Decoder, Encoder, AE
from zsl_error_metric import zsl_error_metric

###torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_printoptions(precision=10)


def r_squared(x, y):
    y_m = np.mean(y)

    r2 = 1 - (np.sum((y-x)*(y-x)) / np.sum((y-y_m)*(y-y_m)))
    if(r2 < 0):
        r2 = 0

    return r2


def func(x, a, m, p):
    return np.cos(m * x + p) + a


def poly4(x, c0, c1, c2, c3, c4):
    return c0*(x**4) + c1*(x**3) + c2*(x**2) + c3*x + c4

def sum_sine_3(x, a1, b1, c1, a2, b2, c2, a3, b3, c3):
    return a1*np.sin(b1*x+c1) + a2*np.sin(b2*x+c2) + a3*np.sin(b3*x+c3)


# def sum_sine_3_bfgs(P, x):
#     res = P[0]*np.sin(P[1]*x+P[2]) + P[3]*np.sin(P[4]*x+P[5]) + P[6]*np.sin(P[7]*x+P[8])
#     return np.sum((res - x)*(res - x))

# def sum_sine3_pso(P, x, y):
#     # p = C.shape[0]
#
#     f_loss = np.empty(0)
#
#     for idx in range(0, P.shape[0]):
#         res = P[idx, 0]*np.sin(P[idx, 1]*x+P[idx, 2]) + P[idx, 3]*np.sin(P[idx, 4]*x+P[idx, 5]) + P[idx, 6]*np.sin(P[idx, 7]*x+P[idx, 8])
#         # f_loss = np.append(f_loss, np.sum((res - Y)*(res - Y)))
#         f_loss = np.append(f_loss, 1-r_squared(res, y))
#
#     return f_loss

# max_epochs = 30000

# number of random samples to generate (should be a multiple of two for flattening an IQ pair)
io_size = 256
feature_size = 1
cluster_size = 8
read_data = True

# setup everything
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# model = AE(io_size, feature_size).to(device)
# decoder = Decoder(feature_size, io_size).to(device)
#
# # this is setup as a static learning rate.  we may want to look at variable lr based on some performance numbers
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.MSELoss()
#
# dec_opt = optim.Adam(decoder.parameters(), lr=1e-3)

if __name__ == '__main__':

    idx = 0
    rng = np.random.default_rng(10)
    # data_bits = 12
    # data_min = 0
    # data_max = 2**data_bits
    # fp_bits = 4

    # features for the decoder
    # F_np = 2048*np.ones(feature_size).astype(np.float32)

    # F = torch.from_numpy((2048*np.ones((1, feature_size))).astype(np.float32)).to(device)
    # F = F.view(-1, feature_size)

    # step 1: load the data
    print("Loading data...\n")

    base_name = "sdr_test"
    iq_data = np.fromfile("../data/" + base_name + "_10M_100m_0001.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)


    # base_name = "VH1-164"
    # xd = np.fromfile("e:/data/zsl/" + base_name + ".sigmf-data.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)

    x_blocks = math.ceil(iq_data.size/io_size)
    data_type = "sdr"

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_name = "fs{:02d}-io{:03d}-".format(feature_size, io_size) + data_type
    log_dir = "../results/" + scenario_name + "/"

    os.makedirs(log_dir, exist_ok=True)

    # writer to save the data
    # test_writer = open((log_dir + base_name + "_{:02d}-bits_M3_".format(fp_bits) + "data.bin"), "wb")


    # # cycle through the decoder side and do whatever needs doing
    # for parameter in model.decoder.parameters():
    #     parameter.requires_grad = False        # uncomment to freeze the training of these layers
    #     p = parameter.data
    #     bp = 0


    print("Processing...\n")

    idx = 50000

    # step 2: the block to process
    y_data = iq_data[idx:(idx + io_size)]

    # step 3: this is the initial weights for the network
    # xw = x / F_np

    # step 4: cluster the data
    # cluster_results = KMeans(n_clusters=cluster_size).fit(xw.reshape(-1, 1))
    #
    # data = xw.reshape(-1, 1)
    # for cluster in range(cluster_size):
    #     t1 = np.float32(cluster_results.labels_ == cluster).reshape(-1, 1)
    #     t2 = 1-t1
    #
    #     data = data*t2
    #     t1 = t1 * cluster_results.cluster_centers_[cluster]
    #     data += t1
    #
    # model.decoder.output_layer.weight.data = nn.Parameter(torch.from_numpy(data.reshape(model.decoder.output_layer.weight.shape)))
    #


    # X = torch.from_numpy(x).to(device)
    # X = X.view(-1, x.size)
    #
    #
    # #
    # # F2 = model.encoder(X)
    #
    # F2 = torch.from_numpy(F_np).to(device)
    # XO2 = model.decoder(F2)


    # dist_mean, dist_std, phase_mean, phase_std = zsl_error_metric(X.numpy(), XO2.detach().numpy())
    #
    # print("dist_mean = {:0.4f}, dist_std = {:0.4f}, phase_mean = {:0.4f}, phase_std = {:0.4f}".format(dist_mean, dist_std, phase_mean, phase_std))

    # try to do curve fitting
    x_data = np.arange(0, io_size, 1)
    y_data = y_data.reshape(-1)

    # numpy fitting


    # pso fitting
    # Set-up hyperparameters
    # pso_options = {'c1': 2.2, 'c2': 2.1, 'w': 1.0}
    # pso_opt = ps.single.GlobalBestPSO(n_particles=200, dimensions=9, options=pso_options)
    # cost, P = pso_opt.optimize(sum_sine3_pso, iters=200, x=c_x, y=xw)
    # print(P)
    #
    x_data_hr = np.arange(0, io_size, 0.01)
    # plt.scatter(c_x, xw, c='blue', s=1, label='data')
    # # plt.plot(c_x, func(c_x, *popt), 'r-', label='fit')
    # # plt.plot(c_x, poly4(c_x, *popt), 'r-', label='fit')
    # plt.plot(c_x2, sum_sine_3(c_x2, *P), 'r-', label='fit')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()


    # scipy fitting
    # popt, pcov = curve_fit(func, c_x, c_y)
    # popt, pcov = curve_fit(poly4, c_x, c_y)
    popt, pcov = curve_fit(sum_sine_3, x_data, y_data)

    print(popt)

    # plt.plot(c_x, c_y, 'b-', label='data')
    plt.scatter(x_data, y_data, c='blue', s=1, label='data')
    # plt.plot(c_x, func(c_x, *popt), 'r-', label='fit')
    # plt.plot(c_x, poly4(c_x, *popt), 'r-', label='fit')
    plt.plot(x_data_hr, sum_sine_3(x_data_hr, *popt), 'r-', label='fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


    # # popt2, pcov2 = curve_fit(func, c_x, xw)
    # # popt2, pcov2 = curve_fit(func, c_x, xw)
    # popt2, pcov2 = curve_fit(sum_sine_3, c_x, xw)
    # print(popt2)
    #
    # plt.scatter(c_x, xw, c='blue', s=1, label='data')
    # # plt.plot(c_x, func(c_x, *popt2), 'r-', label='fit:')
    # # plt.plot(c_x, func(c_x, *popt2), 'r-', label='fit:')
    # plt.plot(c_x2, sum_sine_3(c_x2, *popt2), 'r-', label='fit:')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()


    # bfgs optimization
    # x0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])



    # xopt = fmin_bfgs(sum_sine_3_bfgs, x0, args=(c_x,))
    # print(xopt)
    #
    # plt.scatter(c_x, xw, c='blue', s=1, label='data')
    # # plt.plot(c_x, func(c_x, *popt2), 'r-', label='fit:')
    # # plt.plot(c_x, func(c_x, *popt2), 'r-', label='fit:')
    # plt.plot(c_x2, sum_sine_3_bfgs(xopt, c_x2), 'r-', label='fit:')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()


    # # write the reconstructed data to a binary file
        # t2 = (Y.numpy())[0:x.size]
        # t2 = t2.reshape([x.size]).astype(np.int16)
        # test_writer.write(t2)

    # test_writer.close()


    # just a stopping break point before the code ends
    bp = 9
