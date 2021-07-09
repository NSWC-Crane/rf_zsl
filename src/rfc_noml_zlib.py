'''
This file is an experiment to test compression using a decoder only with fixed point and or integer based weights
'''

import os
import time

import copy
import math
import numpy as np
import datetime

import zlib

import pyswarms as ps

# import the network
from zsl_error_metric import zsl_error_metric


max_epochs = 30000

# number of random samples to generate (should be a multiple of two for flattening an IQ pair)
fft_size = 2**14
io_size = 1024
feature_size = 1
decoder_int1 = 1

read_data = True

# setup everything

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
    data_min = -(2**(data_bits-1))
    data_max = 2**(data_bits-1) - 1
    fp_bits = 4

    # input into the decoder
    F = 2048*np.ones(feature_size).astype(np.float32)

    print("Loading data...\n")

    base_name = "sdr_test"
    xd = np.fromfile("../data/" + base_name + "_10M_100m_0000.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)
    # xd = rng.integers(data_min, data_max, size=(math.floor(fft_size+256)), dtype=np.int16, endpoint=False).astype(np.float32)

    # base_name = "VH1-164"
    # xd = np.fromfile("e:/data/zsl/" + base_name + ".sigmf-data.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)

    x_blocks = math.ceil(xd.size/fft_size)
    data_type = "sdr"

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_name = "fs{:02d}-io{:03d}-".format(feature_size, io_size) + data_type
    log_dir = "../results/" + scenario_name + "/"

    os.makedirs(log_dir, exist_ok=True)

    xd = xd / F

    # compressed_data = zlib.compress(xd, 9)
    compress = zlib.compressobj(zlib.Z_BEST_COMPRESSION, zlib.DEFLATED, +15)
    # compressed_data = compress.compress(xd)
    # compressed_data += compress.flush()

    comp_writer = open((log_dir + base_name + "_{:02d}-bits_M3_".format(fp_bits) + "data.z"), "wb")
    # comp_writer.write(compressed_data)
    # comp_writer.close()


    # writer to save the data
    test_writer = open((log_dir + base_name + "_{:02d}-bits_M3_".format(fp_bits) + "data.bin"), "wb")

    x_f = np.empty([x_blocks, fft_size], dtype=np.float32)
    idx = 0

    for fft_blk in range(0, x_blocks*fft_size, fft_size):
        print("Processing FFT block {:d}".format(fft_blk))
        x = xd[fft_blk:(fft_blk + fft_size)]

        if (x.size < io_size):
            x = np.pad(x, (0, io_size-x.size), 'constant')

        # get the mean of x
        x_mean = math.floor(np.mean(x))
        x_std = np.std(x)

        # convert x into a complex numpy array
        x2 = x.reshape(-1, 2)

        xc = np.empty(x2.shape[0], dtype=complex)
        xc.real = x2[:, 0]
        xc.imag = x2[:, 1]

        # take the FFT of the block
        x_fft = np.fft.fft(xc)/xc.size

        # the FFT version that has been packed into real, imag, real, imag,...
        # x_f = np.zeros([x.size], dtype=np.float32)
        x_f[idx, 0:x.size:2] = x_fft.real
        x_f[idx, 1:x.size:2] = x_fft.imag
        # x_f = np.floor(x_f + 0.5)
        idx = idx + 1

        # container for the reconstructed
        # x_r = np.zeros([x.size], dtype=np.float32)

        # print("Processing...\n")

        # end of for idx in range(0, fft_size, math.ceil(fft_size/io_size))

        # convert the x_r back into the original samples by taking the ifft
        # xr_fft = np.empty(xc.size, dtype=complex)
        # xr_fft.real = x_r[0:x_r.size:2]
        # xr_fft.imag = x_r[1:x_r.size:2]
        #
        # x_i = np.fft.ifft(xr_fft*xr_fft.size)
        #
        # y = np.zeros([x_f.size], dtype=np.float32)
        # y[0:x_f.size:2] = np.floor(x_i.real + 0.5)
        # y[1:x_f.size:2] = np.floor(x_i.imag + 0.5)
        #
        # x = x.reshape(-1)
        # dist_mean, dist_std, phase_mean, phase_std = zsl_error_metric(x, y)
        #
        # print("dist_mean = {:0.4f}, dist_std = {:0.4f}, phase_mean = {:0.4f}, phase_std = {:0.4f}".format(dist_mean, dist_std, phase_mean, phase_std))

        # write the reconstructed data to a binary file
        #t2 = (Y.numpy())[0:xs.size]
        #t2 = t2.reshape([xs.size]).astype(np.int16)
        # test_writer.write(y.astype(np.int16))

    # test_writer.close()

    compressed_data = compress.compress(x_f)
    compressed_data += compress.flush()

    comp_writer = open((log_dir + base_name + "_{:02d}-bits_M3_".format(fp_bits) + "data.z"), "wb")
    comp_writer.write(compressed_data)
    comp_writer.close()

    # just a stopping break point before the code ends
    bp = 9
