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

# from torch.utils.tensorboard import SummaryWriter

# import the network
from zsl_decoder_v1 import Decoder, Encoder, AE
from zsl_error_metric import zsl_error_metric

###torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_printoptions(precision=10)

max_epochs = 30000

# number of random samples to generate (should be a multiple of two for flattening an IQ pair)
io_size = 4096
feature_size = 1
cluster_size = 256
read_data = True

# setup everything
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model = AE(io_size, feature_size).to(device)

# this is setup as a static learning rate.  we may want to look at variable lr based on some performance numbers
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# def process_block():




if __name__ == '__main__':

    idx = 0
    rng = np.random.default_rng(10)
    # data_bits = 12
    # data_min = 0
    # data_max = 2**data_bits
    # fp_bits = 4

    # features for the decoder
    F_np = 2048*np.ones(feature_size).astype(np.float32)

    # F = torch.from_numpy((2048*np.ones((1, feature_size))).astype(np.float32)).to(device)
    # F = F.view(-1, feature_size)

    # step 1: load the data
    print("Loading data...\n")

    base_name = "sdr_test"
    xd = np.fromfile("../data/" + base_name + "_10M_100m_0001.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)


    # base_name = "VH1-164"
    # xd = np.fromfile("e:/data/zsl/" + base_name + ".sigmf-data.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)

    x_blocks = math.ceil(xd.size/io_size)
    data_type = "sdr"

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_name = "fs{:02d}-io{:03d}-".format(feature_size, io_size) + data_type
    log_dir = "../results/" + scenario_name + "/"

    os.makedirs(log_dir, exist_ok=True)



    # writer to save the data
    # test_writer = open((log_dir + base_name + "_{:02d}-bits_M3_".format(fp_bits) + "data.bin"), "wb")


    # cycle through the decoder side and do whatever needs doing
    for parameter in model.decoder.parameters():
        parameter.requires_grad = False        # uncomment to freeze the training of these layers
        p = parameter.data
        bp = 0


    print("Processing...\n")

    idx = 50000

    # step 2: the block to process
    x = xd[idx:(idx + io_size)]

    # step 3: this is the initial weights for the network
    xw = x / F_np

    # step 4: cluster the data
    cluster_results = KMeans(n_clusters=cluster_size).fit(xw.reshape(-1, 1))

    data = xw.reshape(-1, 1)
    for cluster in range(cluster_size):
        t1 = np.float32(cluster_results.labels_ == cluster).reshape(-1, 1)
        t2 = 1-t1

        data = data*t2
        t1 = t1 * cluster_results.cluster_centers_[cluster]
        data += t1

    model.decoder.output_layer.weight.data = nn.Parameter(torch.from_numpy(data.reshape(model.decoder.output_layer.weight.shape)))
    bp = 0



    # for layer in model.decoder.parameters():
    #     data = layer.detach().numpy()
    #     data = data.reshape(-1, 1)
    #
    #     layer.data = nn.Parameter(torch.from_numpy(data.reshape(layer.shape)))
    #     bp = 0



    X = torch.from_numpy(x).to(device)
    X = X.view(-1, x.size)


    # model must be set to train mode
    model.train()

    print("block {:}".format(idx))
    for epoch in range(max_epochs):
        loss = 0
        optimizer.zero_grad()
        XO = model(X)

        train_loss = criterion(XO, X)
        loss += train_loss.item()

        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, loss))

        loss_q = torch.sum(torch.abs(torch.floor(XO + 0.5) - X))
        if (loss < 0.15):
            bp = 10
            break

        train_loss.backward()
        optimizer.step()

    loss = torch.sum(torch.abs(torch.floor(XO + 0.5) - X))
    print("loss = {:.6f}\n".format(loss.item()))


    F2 = model.encoder(X)


    XO2 = model.decoder(F2)


    # for idx in range(0, x_blocks*io_size, io_size):
        # x = xd[idx:(idx + io_size)]
        #
        # # get the mean of x
        # x_mean = math.floor(np.mean(x))
        # x_std = np.std(x)
        #
        # # convert x into a torch tensor variable
        # X = torch.from_numpy(x).to(device)
        # X = X.view(-1, x.size)
        #
        # if (x.size < io_size):
        #     X = torch.nn.functional.pad(X, (0, io_size-x.size))

        # create a data logger to save info on the run
        # data_writer = open((log_dir + scenario_name + "_" + date_time + ".txt"), "w")
        #
        # data_writer.write("#-------------------------------------------------------------------------------\n")
        # data_writer.write("# data bits, min, max:\n{}, {}, {}\n\n".format(data_bits, data_min, data_max))
        # data_writer.write("# io_size:\n{}\n\n".format(io_size))
        # data_writer.write("# feature_size:\n{}\n\n".format(feature_size))
        # data_writer.write("# F:\n")

        # for idx in range(feature_size):
        #     data_writer.write("{:.6f}".format((F.numpy())[0][idx]))
        #     if(idx < feature_size-1):
        #         data_writer.write(", ")
        #     else:
        #         data_writer.write("\n\n")

        # # model must be set to train mode
        # model.train()

        # print("block {:}".format(idx))
        # for epoch in range(max_epochs):
        #     loss = 0
        #     optimizer.zero_grad()
        #     outputs = model(F)
        #
        #     train_loss = criterion(outputs, X)
        #     loss += train_loss.item()
        #
        #     # print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, loss))
        #
        #     loss_q = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
        #     if (loss_q < 1):
        #         bp = 10
        #         break
        #
        #     train_loss.backward()
        #     optimizer.step()
        #
        # loss = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
        # print("loss = {:.6f}\n".format(loss.item()))

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

        # model.eval()
        #
        # # get the weights
        # dw1a = copy.deepcopy(model.output_layer.weight.data).t()
        #
        # # Set-up hyperparameters
        # pso_options = {'c1': 2.1, 'c2': 2.0, 'w': 1.1}
        #
        # time.sleep(1)
        #
        # # -------------------------------------------------------------------------------
        # # run PSO to find the best scale value
        # # -------------------------------------------------------------------------------
        #
        # # for fp_bits in range(8, 13):
        #
        # # the min/max number (0 <= x < fp_range)
        # # fp_range = 2**(fp_bits)      # the max value
        # # fp_min = 0
        # # fp_max = fp_range - 1
        #
        # # the min/max number (fp_range <= x < fp_range-1)
        # fp_range = 2**(fp_bits-1)      # the max value
        # fp_min = -fp_range
        # fp_max = fp_range - 1
        #
        # scale_step = 0.005
        #
        # # min_scale = math.floor(fp_range * 0.25)
        # # max_scale = (fp_range * 0.625) + scale_step
        # # min_loss = 1e10
        # min_scale = (2**(fp_bits-2)) *0.625
        # max_scale = (fp_range * 1.1) + scale_step
        #
        # # scale_bounds = [min_scale, max_scale]
        # scale_bounds = (min_scale*np.ones(1), max_scale*np.ones(1))
        #
        # # data_wr = open((log_dir + scenario_name + "_{:02d}-bits_".format(fp_bits) + date_time + ".txt"), "w")
        #
        # pso_opt = ps.single.GlobalBestPSO(n_particles=40, dimensions=1, options=pso_options, bounds=scale_bounds)
        #
        # # pso_opt.optimize(fx.sphere, iters=100)
        # # cost, scale = pso_opt.optimize(get_best_scale, iters=50, F=F.numpy(), W=dw1a.numpy(), X=X.numpy(), fp_min=fp_min, fp_max=fp_max)
        # scale = [8.0]
        #
        # # data_wr.write("{:0.8f}, {}, ".format(scale[0], cost))
        #
        # dw1a_q = torch.clamp_min(torch.clamp_max(torch.floor(torch.abs(dw1a * scale[0])), fp_max), fp_min)
        # d1 = (torch.sign(dw1a)*dw1a_q)/scale[0]
        #
        # w_q = quantize_weights(dw1a.numpy(), scale[0])
        #
        #
        # # Y = torch.floor((d1.t()*F).sum(axis=1) + 0.5)
        # Y = torch.floor(torch.sum(torch.from_numpy(w_q).t()*F, dim=1) + 0.5)
        # loss2 = torch.sum(torch.abs(Y - X))/X.shape[1]
        #
        # print("scale = {:0.6f}, loss = {:.2f}".format(scale[0], loss2.item()))

        # for idx in range(io_size):
        #     data_wr.write("{}".format((Y.numpy())[idx]))
        #     if(idx<io_size-1):
        #         data_wr.write(", ")
        #     else:
        #         data_wr.write("\n")
        #
        # data_wr.close()

    dist_mean, dist_std, phase_mean, phase_std = zsl_error_metric(X.numpy(), XO2.detach().numpy())

    print("dist_mean = {:0.4f}, dist_abs = {:0.4f}, phase_mean = {:0.4f}, phase_std = {:0.4f}".format(dist_mean, dist_std, phase_mean, phase_std))

        # # write the reconstructed data to a binary file
        # t2 = (Y.numpy())[0:x.size]
        # t2 = t2.reshape([x.size]).astype(np.int16)
        # test_writer.write(t2)

    # test_writer.close()


    # just a stopping break point before the code ends
    bp = 9
