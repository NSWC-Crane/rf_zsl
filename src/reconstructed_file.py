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
import glob

from zsl_error_metric import *
from utils import *

# number of random samples to generate (should be a multiple of two for flattening an IQ pair)
input_size = 2848
max_epochs = 1000
feature_size = 1

n_clusters = 100
read_data = True
device = "cpu"
m = 128

data_file = "lfm_test_10M_100m_0000"

### torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

criterion = nn.MSELoss()


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

if __name__ == '__main__':
    idx = 0
    compressed_loss = 0
    rng = np.random.default_rng()
    lr_shift = 1.0
    
    try: 
        date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_name = "fs{:02}-is{:03}-nc{:03}".format(feature_size, input_size, n_clusters)
        log_dir = "../reconstructed_data/" + scenario_name + "/"

        os.makedirs(log_dir, exist_ok=True)

        files = glob.glob('../data/*.bin')
        for data_file in files:
            data_file  = data_file.split('\\')[-1]
            print(data_file)
            file = open((log_dir + data_file.split('.')[0] + "-" + scenario_name + "-" + date_time + ".bin"), "wb")
            x = np.fromfile("../data/" + data_file, dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)

            if len(x) % input_size != 0:
                print("[WARNING]: Size of data {} is not divisible by input size {}".format(len(x), input_size))
                print("[WARNING]: The last {} entries in file will be dropped".format(len(x) % input_size))

            for idx in range(0, len(x), input_size):
                X = x[np.s_[idx:min(idx + input_size, len(x))]]
                if len(X) == input_size:
                    X = torch.from_numpy(X).to(device)
                    X = X.view(-1, input_size)

                    model = AE(input_size, feature_size).to(device)
                    init_weights(model)
                    model.round_weights(m)

                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)
                    model.train()

                    # ------------------------------------------------------------------------------
                    for epoch in range(max_epochs):
                        model.train()
                        optimizer.zero_grad()
                        outputs = model(X)

                        train_loss = criterion(outputs, X)
                        train_loss.backward()
                        optimizer.step()
                        loss = train_loss.item()

                        if torch.sum(torch.abs(torch.floor(outputs + 0.5) - X)) < 1:
                            break

                        if loss < lr_shift:
                            lr = optimizer.param_groups[0]['lr']
                            optimizer.param_groups[0]['lr'] = 0.95 * lr
                            lr_shift = 0.9 * lr_shift

                    # ------------------------------------------------------------------------------
                    cluster_weights(model, n_clusters)                      # reassign each weight based on its nearest cluster center
                    optimizer = optim.Adam(model.parameters(), lr=5e-3)
                    compressed_loss = criterion(outputs, X).item()
                    [dist_mean, dist_std, phase_mean, phase_std] = zsl_error_metric(X, outputs.detach().numpy())
                    print("step: {}/{}, loss = [{:.6f}, {:.6f},{:.6f},{:.6f},{:.6f}]".format(idx, len(x), compressed_loss, dist_mean, dist_std, phase_mean, phase_std))
            
                file.write(outputs.detach().numpy().astype(np.int16))

            file.close()
    except ValueError:
        print(ValueError)
    
    bp = 0
    input('Program ended. Press Enter to exit...')
