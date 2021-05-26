
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import copy
import math
import numpy as np
import datetime
import os
from sklearn.cluster import KMeans

from model import AE
from utils import *


max_epochs = 800
# number of random samples to generate (should be a multiple of two for flattening an IQ pair)
input_size = 128
feature_size = 16
decoder_int1 = 128
m = 128

read_data = True
device = "cpu"

input_size = [512, 1024]
n_clusters = [2, 4, 6]

###torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# this is setup as a static learning rate.  we may want to look at variable lr based on some performance numbers
#optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)
criterion1 = nn.MSELoss()
criterion2 = DevLoss()
criterion3 = SmallWeightLoss()

#cv2.setNumThreads(0)

#torch.multiprocessing.set_start_method('spawn')
if __name__ == '__main__':

    idx = 0
    rng = np.random.default_rng()

    # model must be set to train mode for QAT logic to work
    # model.train()
    lr_shift = 1.0

    for in_size in input_size:
        for fs in get_feature_size(in_size):
            for di in get_d_init(fs):
                for nc in n_clusters:

                    if (read_data == True):
                        x = np.fromfile("../data/lfm_test_10M_100m_0000.bin", dtype=np.int16, count=-1, sep='',
                                        offset=0).astype(np.float32)
                        x = x[np.s_[idx:idx + in_size]]
                    else:
                        x = rng.integers(-2048, 2048, size=(1, 1, 1, in_size), dtype=np.int16,
                                         endpoint=False).astype(np.float32)

                        # convert x into a torch tensor variable
                    X = torch.from_numpy(x).to(device)
                    X = X.view(-1, in_size)

                    model = AE(in_size, fs, di).to(device)
                    init_weights(model)

                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)

                    model.train()

                    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    scenario_name = "fs{:02}-is{:03}".format(fs, in_size)
                    log_dir = "../results/" + scenario_name + "/"

                    os.makedirs(log_dir, exist_ok=True)

                    data_writer = open((log_dir + scenario_name + "_" + date_time + ".txt"), "w")
                    data_writer.write("#-------------------------------------------------------------------------------\n")

                    data_writer.write("input_size: {}\n".format(in_size))
                    data_writer.write("feature_size: {}\n".format(fs))
                    data_writer.write("decoder_init1: {}\n".format(di))
                    data_writer.write("n_clusters: {}\n".format(nc))
                    data_writer.write("m: {}\n".format(m))

                    model.round_weights(m)
                    outputs = model(X)
                    loss = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
                    print("\nloss = {:.6f}".format(loss.item()))
                    data_writer.write("Init_abs_error: {}\n".format(loss))

                    print(log_dir)

                    for epoch in range(max_epochs):
                        model.train()
                        loss = 0
                        optimizer.zero_grad()
                        outputs = model(X)

                        train_loss = criterion1(outputs, X)
                        train_loss.backward()
                        optimizer.step()
                        loss += train_loss.item()

                        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, loss))

                        if (torch.sum(torch.abs(torch.floor(outputs + 0.5) - X)) < 1):
                            break

                        if (loss < lr_shift):
                            lr = optimizer.param_groups[0]['lr']
                            optimizer.param_groups[0]['lr'] = 0.95 * lr
                            lr_shift = 0.9 * lr_shift

                    print(f"input_size:{in_size}-feature_size:{fs}-decoder_int1:{di}-n_clusters:{nc}")
                    cluster_weights(model, nc)

                    outputs = model(X)
                    loss = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
                    print("\nloss = {:.6f}".format(loss.item()))

                    optimizer = optim.Adam(model.parameters(), lr=5e-3)
                    model.freeze_decoder()

                    max_epochs = 2000

                    ## train encoder once more
                    for epoch in range(max_epochs):
                        model.train()
                        loss = 0
                        optimizer.zero_grad()
                        outputs = model(X)

                        train_loss = criterion1(outputs, X)
                        train_loss.backward()
                        optimizer.step()
                        loss += train_loss.item()

                        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, loss))

                        if (torch.sum(torch.abs(torch.floor(outputs + 0.5) - X)) < 1):
                            break

                        if (loss < lr_shift):
                            lr = optimizer.param_groups[0]['lr']
                            optimizer.param_groups[0]['lr'] = 0.95 * lr
                            lr_shift = 0.9 * lr_shift

                    data_writer.write("Final_MSE: {}\n".format(loss))

                    loss = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
                    print("\nloss = {:.6f}\n".format(loss.item()))

                    data_writer.write("Final_abs_error: {}\n".format(loss))
                    data_writer.close()

                    do_some_debug_stuff = 0

    do_some_debug_stuff = 0




