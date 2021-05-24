
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
from params import *
from utils import *

###torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# setup everything
model = AE(input_size, feature_size).to(device)
init_weights(model)

# this is setup as a static learning rate.  we may want to look at variable lr based on some performance numbers
#optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)
criterion1 = nn.MSELoss()
criterion2 = DevLoss()
criterion3 = SmallWeightLoss()

#cv2.setNumThreads(0)

#torch.multiprocessing.set_start_method('spawn')
if __name__ == '__main__':

    idx = 0
    rng = np.random.default_rng()

    if(read_data == True):
        x = np.fromfile("../data/lfm_test_10M_100m_0000.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)
        x = x[np.s_[idx:idx+input_size]]
    else:
        x = rng.integers(-2048, 2048, size=(1, 1, 1, input_size), dtype=np.int16, endpoint=False).astype(np.float32)

    # convert x into a torch tensor variable
    X = torch.from_numpy(x).to(device)
    X = X.view(-1, input_size)

    # model must be set to train mode for QAT logic to work
    model.train()
    lr_shift = 1.0
    frozen_encoder = False
    data_type = 'unfrozen-binary'

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_name = "fs{:02}-m{:03}-".format(feature_size, m) + data_type
    log_dir = "../results/" + scenario_name + "/"

    os.makedirs(log_dir, exist_ok=True)

    data_writer = open((log_dir + scenario_name + "_" + date_time + ".txt"), "w")
    data_writer.write("#-------------------------------------------------------------------------------\n")

    data_writer.write("feature_size: {}\n".format(feature_size))
    data_writer.write("m: {}\n".format(m))
    data_writer.write("update_weights: {}\n".format(update_weights))
    data_writer.write("frozen_encoder: {}\n".format(frozen_encoder))

    model.round_weights(m)
    outputs = model(X)
    loss = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
    print("\nloss = {:.6f}".format(loss.item()))

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


    cluster_weights(model, 4)

    outputs = model(X)
    loss = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
    print("\nloss = {:.6f}".format(loss.item()))

    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    model.freeze_decoder()

    max_epochs *= 3

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

    loss = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
    print("\nloss = {:.6f}".format(loss.item()))

    data_writer.write("Final_Loss: {}\n".format(loss))
    data_writer.close()

    do_some_debug_stuff = 0
