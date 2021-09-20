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

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy import diff
from sklearn.metrics import r2_score
from numpy.random import rand

from zsl_error_metric import *
from utils import *
from zsl_clustering import *

# number of random samples to generate (should be a multiple of two for flattening an IQ pair)
input_size = 1024
max_epochs = 1000
feature_size = 1
degree = 2

num_clusters_max = 50
num_clusters_step = 50
read_data = True
device = "cpu"
m = 128

final_zsl_metric = ()
init_mse_loss = 0
best_mse_loss = 0

dist_mean_arr = []
dist_std_arr = []
phase_mean_arr = []
phase_std_arr = []

loss_arr = []

data_file = "lfm_test_10M_100m_0000"

r = rand()


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
        reconstructed = self.input_layer(activation)
        return reconstructed


# use the encoder and decoder classes to build the autoencoder
class AE(nn.Module):
    def __init__(self, input_size, feature_size):
        super().__init__()
        self.encoder = Encoder(input_size, feature_size)
        self.decoder = Decoder(input_size, feature_size)

    def forward(self, features):
        code = self.encoder(features)
        code.to(device)
        reconstructed = self.decoder(code)
        return reconstructed

    def get_encoder(self):
        return self.encoder

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

    iidx = 1563521
    rng = np.random.default_rng()

    # model must be set to train mode for QAT logic to work
    # model.train()
    lr_shift = 1.0

    for n_clusters in [100, 100]:
        if read_data == True:
            x = np.fromfile("../data/" + data_file + ".bin", dtype=np.int16, count=-1, sep='',
                            offset=0).astype(np.float32)
            x = x[np.s_[iidx:iidx + input_size]]
        else:
            x = rng.integers(-2048, 2048, size=(1, 1, 1, input_size), dtype=np.int16,
                             endpoint=False).astype(np.float32)

        # convert x into a torch tensor variable
        X = torch.from_numpy(x).to(device)
        X = X.view(-1, input_size)

        bu = torch.from_numpy(x).to(device)
        bu = bu.view(-1, input_size)

        plot_data(X.detach().numpy())

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

            [dist_mean, dist_std, phase_mean, phase_std] = zsl_error_metric(X, outputs.detach().numpy())
            dist_mean_arr.append(dist_mean)
            dist_std_arr.append(dist_std)
            phase_mean_arr.append(phase_mean)
            phase_std_arr.append(phase_std)

            loss_arr.append(loss)

            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, loss))

            if torch.sum(torch.abs(torch.floor(outputs + 0.5) - X)) < 1:
                break

            if loss < lr_shift:
                lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = 0.95 * lr
                lr_shift = 0.9 * lr_shift

        # ------------------------------------------------------------------------------
        # functions for plotting dist_mean_arr, dist_std_arr, phase_mean_arr, phase_std_arr, loss_arr

        # reset variables to empty arrays for next for loop iteration (different n_clusters)
        dist_mean_arr = []
        dist_std_arr = []
        phase_mean_arr = []
        phase_std_arr = []

        loss_arr = []

        # ------------------------------------------------------------------------------
        # reassign each weight based on its nearest cluster center
        labels, centers = cluster_weights(model, n_clusters, cluster_name='KMeans')
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

            [dist_mean, dist_std, phase_mean, phase_std] = zsl_error_metric(X, outputs.detach().numpy())
            print("epoch: {}/{}, loss = [{:.6f}, {:.6f},{:.6f},{:.6f},{:.6f}]"
                  .format(epoch + 1, max_epochs, loss, dist_mean, dist_std, phase_mean, phase_std))

            if loss < best_mse_loss:
                best_mse_loss = loss
                final_zsl_metric = zsl_error_metric(X, outputs.detach().numpy())

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

        # ------------------------------------------------------------------------------

        # save encoded weights (labels) to .bin file and read from it
        # weights_writer = open((log_dir + scenario_name + "-" + 'TEST_weights' + "-" + date_time + ".bin"), "wb")
        # model_weights = model.decoder.input_layer.weight.detach().numpy()
        # weights_writer.write(model_weights.astype(np.float32))
        # weights_writer.close()
        #
        # read_weights = np.fromfile(log_dir + scenario_name + "-" + 'TEST_weights' + "-" + date_time + ".bin",
        #                            dtype=np.float32, count=-1, sep='', offset=0).astype(np.float32)
        #
        # label_writer = open((log_dir + scenario_name + "-" + 'TEST_labels' + "-" + date_time + ".bin"), "wb")
        # label_writer.write(labels.astype(np.uint8))
        # label_writer.close()
        #
        # read_labels = np.fromfile(log_dir + scenario_name + "-" + 'TEST_labels' + "-" + date_time + ".bin",
        #                           dtype=np.uint8, count=-1, sep='', offset=0).astype(np.float32)

        bp = 0
        model_v2 = AE(input_size, feature_size)
        model_v2.encoder = model.get_encoder()
        # new_model.decoder.input_layer.weight.data = nn.Parameter(torch.from_numpy(read_labels.reshape(model.decoder.input_layer.weight.shape)))

        # read_labels = read_labels.reshape(-1, 1)
        # converted_weights = assign_weights(read_labels, centers)
        #
        # converted_weights = converted_weights.astype(np.float32)
        # converted_weights = nn.Parameter(
        #     torch.from_numpy(converted_weights.reshape(model.decoder.input_layer.weight.shape)))
        # model_v2.decoder.input_layer.weight = converted_weights
        #
        new_outputs = model_v2(X)

        print(f'Reconstruction Loss: {criterion(new_outputs, X).item()}')

        X_ = np.array(range(0, n_clusters)).reshape(-1, 1)
        y = np.sort(centers, axis=0)

        t = np.sort(centers, axis=0)
        idx = centers.argsort(axis=0)

        print((centers[idx.reshape(n_clusters, )] == t).sum())

        a = find_discontinuity_points(y, n_clusters=n_clusters)
        temp = get_ranges(a, last_idx=1023)
        models = get_models(X_, y, temp, degree=degree)

        sorted_labels = sort_labels(labels, idx.reshape(n_clusters, ))
        print(f'Min test: {centers[idx[0]].item() == centers.min()}')
        print(f'Max test: {centers[idx[-1]].item() == centers.max()}')

        # testing the labels and models ability
        # TODO: convert below to work with polynomial features, if degree exists then create features
        # rand_idx = int(np.floor(r * 1024))
        # test_idx = search_range(labels[rand_idx], temp)
        # print(models[test_idx].coef_ * labels[rand_idx] + models[test_idx].intercept_)
        # print(f'Original value: {y[labels[rand_idx]]}')

        # use models to replace all weights
        holder = np.copy(labels).astype(np.float32)
        for i, label in enumerate(labels):
            j = search_range(label, temp)
            # holder[i] = models[j].coef_ * label + models[j].intercept_
            poly = PolynomialFeatures(degree=degree)
            features = poly.fit_transform(np.int32(label).reshape(-1, 1))
            holder[i] = models[j].predict(features)

        model_v2.unfreeze_decoder()
        original_weights = model_v2.decoder.input_layer.weight.data
        shape = original_weights.numpy().shape
        model_v2.decoder.input_layer.weight.data = nn.Parameter(torch.from_numpy(holder.reshape(shape)))
        model_v2.freeze_decoder()

        outputs = model_v2(X)
        train_loss = criterion(outputs, X).item()
        print(f'[Model_v2 LOSS]: {train_loss}')

        d1 = model.decoder.input_layer.weight.data
        d2 = model_v2.decoder.input_layer.weight.data
        x_t = np.array(range(d1.size()[0]))

        plt.figure()
        plt.scatter(x_t, d1, color='red', label='original')
        plt.scatter(x_t, d2, color='blue', label='new')
        plt.plot(x_t, d1, color='red')
        plt.plot(x_t, d2, color='blue')
        plt.legend()
        plt.show()

        bp = 0


bp = 0
