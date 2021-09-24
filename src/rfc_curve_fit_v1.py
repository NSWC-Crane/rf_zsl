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
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.set_printoptions(precision=10)

function_terms = 5

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

def sum_sine_8(x, a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4, a5, b5, c5, a6, b6, c6, a7, b7, c7, a8, b8, c8):
    return a1*np.sin(b1*x+c1) + a2*np.sin(b2*x+c2) + a3*np.sin(b3*x+c3) + a4*np.sin(b4*x+c4) + a5*np.sin(b5*x+c5) + a6*np.sin(b6*x+c6) + a7*np.sin(b7*x+c7) + a8*np.sin(b8*x+c8)


def sum_sine(x, *p):
    s = 0
    for idx in range(0, function_terms):
        s += p[3*idx] * np.sin(p[3*idx+1]*x + p[3*idx+2])

    return s

def sum_fourier(x, *p):
    s = p[0]
    w = p[function_terms-1]

    for idx in range(1, function_terms-1):
        s += p[2*idx+1] * np.cos((idx+1)*w*x) + p[2*idx+2]*np.sin((idx+1)*w*x)

    return s


def get_ssin_start(n, x_data, y_data):
    y0 = np.ones([n * 3], dtype=np.float32)
    x0 = np.zeros([y_data.size, 2 * n], dtype=np.float32)

    peaks = []

    res = y_data

    for jdx in range(0, n, 1):
        fy = np.fft.fft(res)
        fy[peaks] = 0

        ml = np.argmax(np.abs(fy[0:math.floor(y_data.size / 2)]))
        peaks.append(ml)

        y0[3 * jdx + 1] = 2 * np.pi * (max(0.5, ml)) / (x_data[-1] - x_data[0])

        x0[:, 2 * jdx] = np.sin(y0[3 * jdx + 1] * x_data)
        x0[:, 2 * jdx + 1] = np.cos(y0[3 * jdx + 1] * x_data)

        ab = np.matmul(np.linalg.pinv(x0[:, 0:2 * jdx+2]), y_data).reshape(-1)

        y0[3 * jdx] = math.sqrt(ab[2*jdx]*ab[2*jdx] + ab[2*jdx+1]*ab[2*jdx+1])
        y0[3 * jdx + 2] = math.atan2(ab[2*jdx+1], ab[2*jdx])

        if jdx < (n-1):
            res = y_data - np.matmul(x0[:, 0:2 * jdx+2], ab)

    return y0

#------------------------------------------------------------------------------

max_epochs = 30000

# number of random samples to generate (should be a multiple of two for flattening an IQ pair)
# io_size = 128
# feature_size = 1
# cluster_size = 8
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

    # base_name = "lfm_test"
    # base_name = "rand_test"
    base_name = "sdr_test"
    iq_data = np.fromfile("../data/" + base_name + "_10M_100m_0000.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)

    # base_name = "VH1-164"
    # iq_data = np.fromfile("e:/data/zsl/" + base_name + ".sigmf-data.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)

    # y_blocks = math.ceil(iq_data.size/io_size)
    data_type = "sdr"

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_name = "sin3_io_cf_" + data_type
    log_dir = "../results/" + scenario_name + "/"

    os.makedirs(log_dir, exist_ok=True)

    # writer to save the data
    test_writer = open((log_dir + base_name + "_recon_data_" + date_time + ".bin"), "wb")

    # writer to save the results of each block's compression
    data_wr = open((log_dir + base_name + "_recon_results_" + date_time + ".txt"), "w")

    # test writer for the binary zip file
    zip_data = open((log_dir + base_name + "_comp_" + date_time + ".zsl"), "wb")

    print("Processing...\n")
    min_exp = math.ceil(math.log10(function_terms*3*4)/math.log10(2))+1
    max_exp = max(7, min_exp)
    io_size_list = 2**np.arange(max_exp, min_exp-1, -1)
    # io_size_list = 2**np.arange(max_exp, max_exp-1, -1)

    print("Function terms: {:}\nmin exp: {:}\nmax exp: {:}\nio list: {:}\n".format(function_terms, min_exp, max_exp, io_size_list))
    data_wr.write("# Function terms: {:}\n# min exp: {:}\n# max exp: {:}\n# io list: {:}\n".format(function_terms, min_exp, max_exp, io_size_list))

    # index into the data file
    file_index = 0

    # counter for the number of bytes used to compress files
    comp_bytes = 0

    # r-squared fit value that is considered good
    r2_fit = 0.99

    barLength = 20

    bounds = ([-np.inf, 0, -np.inf]*function_terms, np.inf)

    # loop that runs through the data file in io_size_list[...] increments
    # TODO
    while(file_index < iq_data.size):
        size_index = 0

        # this check is to see if the remaining data in the file is large enough to process
        if (iq_data.size - file_index) < io_size_list[0]:
            for size_index in range(0, io_size_list.size, 1):
                if (iq_data.size - (file_index + io_size_list[size_index])) > 0:
                    break

        # loop through each of the io_size_list values and stop when a good fit is reached
        for idx in range(size_index, io_size_list.size, 1):

            # grab a chunk of data
            y_data = iq_data[file_index:(file_index + io_size_list[idx])].reshape(-1)
            x_data = np.arange(0, y_data.size, 1)

            try:
                # create initial guess based on frequency content
                y0 = get_ssin_start(function_terms, x_data, y_data)

                # run the fit based on the data chunk and the initial guess
                fit_values, fit_cov = curve_fit(f=sum_sine, xdata=x_data, ydata=y_data, p0=y0, method='trf', bounds=bounds)

                # get the reconstructed values based on the fit values
                y_hat = sum_sine(x_data, *fit_values)
                y_hat = np.floor(y_hat + 0.5)
                r2 = r2_score(y_data, y_hat)

                if r2 > r2_fit:
                    # increment the compression byte counter: number of coefficients * size of float
                    comp_bytes += (function_terms * 3) * 4 + 1

                    #increment the file_counter
                    file_index += y_data.size

                    # calculate the metrics and print
                    dist_mean, dist_std, phase_mean, phase_std = zsl_error_metric(y_data, y_hat)
                    # print("block {:}: dist_mean = {:0.4f}, dist_abs = {:0.4f}, phase_mean = {:0.4f}, phase_std = {:0.4f}, r_squared = {:0.4f}".format(
                    #         file_index, dist_mean, dist_std, phase_mean, phase_std, r2))
                    # print(".", end='')

                    # save the results to the file
                    data_wr.write("{:},{:},{:0.5f},{:0.5f},{:0.5f},{:0.5f},{:0.4f}\n".format(file_index, y_data.size, dist_mean, dist_std, phase_mean, phase_std, r2))

                    # save the values to the file
                    test_writer.write(y_hat.astype(np.int16))
                    zip_data.write(fit_values.astype(np.float32))

                    break

                elif idx == io_size_list.size-1:
                    y_hat = y_data

                    # increment the compression byte counter
                    comp_bytes += y_data.size * 2 + 1

                    # increment the file_counter
                    file_index += y_data.size

                    r2 = 1

                    # calculate the metrics and print
                    dist_mean, dist_std, phase_mean, phase_std = zsl_error_metric(y_data, y_hat)
                    # print("block {:}: dist_mean = {:0.4f}, dist_abs = {:0.4f}, phase_mean = {:0.4f}, phase_std = {:0.4f}, r_squared = {:0.4f}".format(
                    #         file_index, dist_mean, dist_std, phase_mean, phase_std, r2))
                    # print(".", end='')

                    # save the results to the file
                    data_wr.write("{:},{:},{:0.5f},{:0.5f},{:0.5f},{:0.5f},{:0.4f}\n".format(file_index, y_data.size, dist_mean, dist_std, phase_mean, phase_std, r2))

                    # save the values to the file
                    test_writer.write(y_hat.astype(np.int16))
                    zip_data.write(y_hat.astype(np.int16))

            except RuntimeError:
                # print("No good fit!")

                if idx == io_size_list.size - 1:
                    y_hat = y_data

                    # increment the compression byte counter
                    comp_bytes += y_data.size * 2 + 1

                    # increment the file_counter
                    file_index += y_data.size

                    r2 = 1

                    # calculate the metrics and print
                    dist_mean, dist_std, phase_mean, phase_std = zsl_error_metric(y_data, y_hat)
                    # print("block {:}: dist_mean = {:0.4f}, dist_abs = {:0.4f}, phase_mean = {:0.4f}, phase_std = {:0.4f}, r_squared = {:0.4f}".format(
                    #         file_index, dist_mean, dist_std, phase_mean, phase_std, r2))
                    # print(".", end='')

                    # save the results to the file
                    data_wr.write("{:},{:},{:0.5f},{:0.5f},{:0.5f},{:0.5f},{:0.4f}\n".format(file_index, y_data.size, dist_mean, dist_std, phase_mean, phase_std, r2))

                    # save the values to the file
                    test_writer.write(y_hat.astype(np.int16))
                    zip_data.write(y_hat.astype(np.int16))

                    break

        progress = file_index/iq_data.size
        block = int(round(barLength * progress))
        text = "\rPercent: [{:}] {:5.3f}%, ratio = {:0.6f}".format("#" * block + "-" * (barLength - block), progress * 100, 1-comp_bytes/(file_index*2))
        print(text, end='')

    test_writer.close()
    zip_data.close()

    print("\n\nbytes Processed = {:}, bytes stored = {:}, ratio = {:0.6f}".format(iq_data.size*2, comp_bytes, 1-comp_bytes/(iq_data.size*2)))

    data_wr.write("\n#bytes Processed = {:}, bytes stored = {:}, ratio = {:0.6f}\n".format(iq_data.size*2, comp_bytes, 1-comp_bytes/(iq_data.size*2)))
    data_wr.close()

    # just a stopping break point before the code ends
    bp = 9
