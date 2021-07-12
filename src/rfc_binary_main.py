import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import copy
import math
import torch.optim as optim
import datetime
import os

from zsl_error_metric import *
from utils import *

# number of random samples to generate (should be a multiple of two for flattening an IQ pair)
input_size = 1024
max_epochs = 10000
feature_size = 1

num_clusters_max = 50
num_clusters_step = 50
read_data = True
device = "cpu"
m = 128

final_zsl_metric = ()
init_mse_loss = 0
best_mse_loss = 0

data_file = "lfm_test_10M_100m_0000"


# create the encoder class
class Encoder(nn.Module):
    def __init__(self, input_size, feature_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 16, bias=False)
        self.hidden_layer_1 = nn.Linear(16, 128, bias=False)
        self.hidden_layer_2 = nn.Linear(128, 256, bias=False)
        self.output_layer = nn.Linear(256, feature_size, bias=False)

    def forward(self, activation):
        activation = self.input_layer(activation)
        activation = self.hidden_layer_1(activation)
        activation = self.hidden_layer_2(activation)
        activation = self.output_layer(activation)
        return activation


# create the decoder class
class Decoder(nn.Module):
    def __init__(self, output_size, feature_size):
        super().__init__()
        self.input_layer = nn.Linear(feature_size, output_size, bias=False)

    def forward(self, activation):
        activation = self.input_layer(activation)
        return activation


# use the encoder and decoder classes to build the autoencoder
class AE(nn.Module):
    def __init__(self, input_size, feature_size):
        super().__init__()
        self.encoder = Encoder(input_size, feature_size)
        self.decoder = Decoder(input_size, feature_size)

    def forward(self, features):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed

    def round_weights(self, m):
        with torch.no_grad():
            for param in self.decoder.parameters():
                t1 = param.data
                t1 = 2 * (t1 > 0).type(torch.float32) - 1
                t1 = t1 / m
                param.data = t1

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def unfreeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = True


### torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# this is setup as a static learning rate.  we may want to look at variable lr based on some performance numbers
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)
criterion = nn.MSELoss()

# cv2.setNumThreads(0)
# torch.multiprocessing.set_start_method('spawn')
if __name__ == '__main__':

    idx = 0
    rng = np.random.default_rng()

    # model must be set to train mode for QAT logic to work
    # model.train()
    lr_shift = 1.0

    for n_clusters in [*range(1, 3), *range(10, num_clusters_max+num_clusters_step, num_clusters_step)]:
        if read_data == True:
            x = np.fromfile("../data/" + data_file + ".bin", dtype=np.int16, count=-1, sep='',
                            offset=0).astype(np.float32)
            x = x[np.s_[idx:idx + input_size]]
        else:
            x = rng.integers(-2048, 2048, size=(1, 1, 1, input_size), dtype=np.int16,
                             endpoint=False).astype(np.float32)

        # convert x into a torch tensor variable
        X = torch.from_numpy(x).to(device)
        X = X.view(-1, input_size)

        model = AE(input_size, feature_size).to(device)
        init_weights(model)
        model.round_weights(m)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)
        model.train()

        date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_name = "fs{:02}-is{:03}".format(feature_size, input_size)
        log_dir = "../results/" + scenario_name + "/"

        os.makedirs(log_dir, exist_ok=True)

        data_writer = open((log_dir + scenario_name + "_" + date_time + ".txt"), "w")
        data_writer.write("#-------------------------------------------------------------------------------\n")

        data_writer.write("data_file: {}\n".format(data_file))
        data_writer.write("input_size: {}\n".format(input_size))
        data_writer.write("feature_size: {}\n".format(feature_size))
        data_writer.write("n_clusters: {}\n".format(n_clusters))
        data_writer.write("m: {}\n".format(m))

        print("#-------------------------------------------------------------------------------\n")
        print(log_dir)
        print(f"input_size:{input_size}-feature_size:{feature_size}-n_clusters:{n_clusters}")

        # ------------------------------------------------------------------------------
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X)

            train_loss = criterion(outputs, X)
            train_loss.backward()
            optimizer.step()
            loss = train_loss.item()

            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, loss))

            if torch.sum(torch.abs(torch.floor(outputs + 0.5) - X)) < 1:
                break

            if loss < lr_shift:
                lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = 0.95 * lr
                lr_shift = 0.9 * lr_shift

        # ------------------------------------------------------------------------------
        # reassign each weight based on its nearest cluster center
        cluster_weights(model, n_clusters)
        optimizer = optim.Adam(model.parameters(), lr=5e-3)
        model.freeze_decoder()

        # calculate and save initial loss/error metrics
        outputs = model(X)
        init_abs_loss = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
        init_mse_loss = criterion(outputs, X)
        init_zsl_metrics = zsl_error_metric(x, outputs.detach().numpy())

        best_mse_loss = init_mse_loss

        # data_writer.write("Init_abs_loss: {}\n".format(init_abs_loss))
        data_writer.write("Init_mse_loss: {}\n".format(init_mse_loss))
        data_writer.write("init_zsl_metrics: {}\n".format(",".join(str(x) for x in init_zsl_metrics)))

        # ------------------------------------------------------------------------------
        # optimize encoder with frozen decoder
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X)

            train_loss = criterion(outputs, X)
            train_loss.backward()
            optimizer.step()
            loss = train_loss.item()

            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, loss))

            if loss < best_mse_loss:
                best_mse_loss = loss
                final_zsl_metric = zsl_error_metric(X, outputs.detach().numpy())
                torch.save(model.state_dict(), "../nets/" + data_file + "_" + scenario_name + "_" + date_time + ".pth")

            if torch.sum(torch.abs(torch.floor(outputs + 0.5) - X)) < 1:
                break

            if loss < lr_shift:
                lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = 0.95 * lr
                lr_shift = 0.9 * lr_shift

        data_writer.write("best_mse_loss: {}\n".format(best_mse_loss))
        data_writer.write("final_zsl_metrics: {}\n".format(",".join(str(x) for x in final_zsl_metric)))
        data_writer.close()

        bp = 0

    bp = 0

