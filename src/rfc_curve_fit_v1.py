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
from sklearn.metrics import r2_score

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

def sum_sine_4(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4):
    return a1*np.sin(b1*x+c1) + a2*np.sin(b2*x+c2) + a3*np.sin(b3*x+c3) + a4*np.sin(b4*x+c4)

max_epochs = 30000

# number of random samples to generate (should be a multiple of two for flattening an IQ pair)
io_size = 128
feature_size = 1
cluster_size = 8
read_data = True

if __name__ == '__main__':

    idx = 0
    rng = np.random.default_rng(10)
    # data_bits = 12
    # data_min = 0
    # data_max = 2**data_bits
    # fp_bits = 4

    # step 1: load the data
    print("Loading data...\n")

    base_name = "sdr_test"
    iq_data = np.fromfile("../data/" + base_name + "_10M_100m_0001.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)


    # base_name = "VH1-164"
    # xd = np.fromfile("e:/data/zsl/" + base_name + ".sigmf-data.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)

    y_blocks = math.ceil(iq_data.size/io_size)
    data_type = "sdr"

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_name = "sin3_io{:03d}_".format(io_size) + data_type
    log_dir = "../results/" + scenario_name + "/"

    os.makedirs(log_dir, exist_ok=True)

    # writer to save the data
    test_writer = open((log_dir + base_name + "_reconstructed_data.bin"), "wb")

    print("Processing...\n")

    idx = 50000
    for idx in range(0, y_blocks*io_size, io_size):

        # step 1: the block to process
        y_data = iq_data[idx:(idx + io_size)].reshape(-1)
        x_data = np.arange(0, y_data.shape[0], 1)
        x_data_hr = np.arange(0, y_data.shape[0], 0.01)

        sine_size = 4

        # create initial guess based on frequency content
        x0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        y0 = np.ones([sine_size*3], dtype=np.float32)

        fy = np.fft.fft(y_data)
        for jdx in range(0, sine_size, 1):
            y0[3*jdx] = np.mean(np.abs(y_data))/(jdx+1)

            ml = np.argmax(np.abs(fy[0:math.floor(io_size/2)]))
            y0[3*jdx+1] = 2 * np.pi * (max(0.5, ml)) / (x_data[-1] - x_data[0])
            fy[ml] = 0

            bp = 0

        try:
            # scipy fitting
            popt, pcov = curve_fit(f=sum_sine_4, xdata=x_data, ydata=y_data, p0=y0)


            # plt.scatter(x_data, y_data, c='blue', s=1, label='data')
            # plt.plot(x_data_hr, sum_sine_3(x_data_hr, *popt), 'r-', label='fit')
            # plt.xlabel('x')
            # plt.ylabel('y')
            # plt.legend()
            # plt.show()

            # get the reconstructed value
            y_hat = np.floor(sum_sine_4(x_data, *popt) + 0.5)

            # print(popt)
            #
        except RuntimeError:
            print("RuntimeError")

            y_hat = y_data

        # calculate the metrics
        dist_mean, dist_std, phase_mean, phase_std = zsl_error_metric(y_data, y_hat)
        print("block {:}: dist_mean = {:0.4f}, dist_abs = {:0.4f}, phase_mean = {:0.4f}, phase_std = {:0.4f}".format(idx, dist_mean, dist_std, phase_mean, phase_std))

        print("r_squared = {:0.4f}\n".format(r2_score(y_data, y_hat)))

        bp = 1

        # write the reconstructed data to a binary file
        # # t2 = (Y.numpy())[0:x.size]
        # # t2 = y_hat.astype(np.int16)
        test_writer.write(y_hat.astype(np.int16))

    test_writer.close()


    # just a stopping break point before the code ends
    bp = 9
