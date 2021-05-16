'''
This file is an experiment to test compression using a decoder only with fixed point and or integer based weights
'''

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import copy
import math
import numpy as np
import datetime

from torch.utils.tensorboard import SummaryWriter


###torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_printoptions(precision=10)

max_epochs = 20000

# number of random samples to generate (should be a multiple of two for flattening an IQ pair)
io_size = 16
feature_size = 1
decoder_int1 = 1

read_data = False

class scale_net(nn.Module):
    def __init__(self, fp_max, fp_min):
        super(scale_net, self).__init__()
        # self.scale = nn.Parameter(data=torch.tensor(1.0), requires_grad=True)
        self.scale = torch.nn.Parameter(torch.randn(()))

    # self.scale = nn.Linear(1, 1, bias=False)
        self.fp_max = fp_max
        self.fp_min = fp_min

    def forward(self, feature, dec):
        d_ol = copy.deepcopy(dec.output_layer.weight)

        # d_ol_q = torch.clamp_min(torch.clamp_max(torch.floor(self.scale(d_ol)), self.fp_max), self.fp_min)
        # dec.output_layer.weight.data = torch.floor(d_ol_q + 0.5)/self.scale.weight.data

        d_ol_q = torch.clamp_min(torch.clamp_max(torch.floor(d_ol * self.scale), self.fp_max), self.fp_min)
        dec.output_layer.weight.data = torch.floor(d_ol_q + 0.5)/self.scale

        xh = torch.floor(dec(feature) + 0.5)
        return xh


# create the decoder class
class Decoder(nn.Module):
    def __init__(self, output_size, feature_size):
        super().__init__()
        self.input_layer = nn.Linear(feature_size, decoder_int1, bias=False)
        #self.hidden_layer_1 = nn.Linear(decoder_int1, 64, bias=False)
        #self.hidden_layer_2 = nn.Linear(64, decoder_int1, bias=False)
        self.output_layer = nn.Linear(decoder_int1, output_size, bias=False)
        #self.prelu = nn.PReLU(1, 0.25)
        #self.multp = nn.Parameter(torch.tensor([[2048.0]]))
        #self.silu = nn.SiLU()
        #self.elu = nn.ELU()
        #self.tanshrink = nn.Tanhshrink()
        #self.tanh = nn.Tanh()
        #self.relu = nn.ReLU()
        #self.alpha = nn.Parameter(torch.tensor(10.0, requires_grad=True))

    def forward(self, activation):
        #activation = self.prelu(activation)
        #activation = self.input_layer(activation)
        #activation = self.elu(activation)
        #activation = self.prelu(activation)
        #activation = self.relu(activation)
        #activation = activation*self.multp.expand_as(activation)
        #activation = self.hidden_layer_1(activation)
        #activation = self.hidden_layer_2(activation)
        #activation = self.silu(activation)
        activation = self.output_layer(activation)
        #activation = self.tanh(activation)
        #activation = self.prelu(activation)
        #activation = self.silu(activation)
        #activation = self.elu(activation)
        #activation = activation
        return activation

# use the encoder and decoder classes to build the autoencoder
class AE(nn.Module):
    def __init__(self, io_size, feature_size):
        super().__init__()
        self.decoder = Decoder(io_size, feature_size)

    def forward(self, features):
        reconstructed = self.decoder(features)
        return reconstructed

# setup everything
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model = AE(io_size, feature_size).to(device)

# cycle through the decoder side and do whatever needs doing
for parameter in model.decoder.parameters():
    #parameter.requires_grad = False        # uncomment to freeze the training of these layers
    p = parameter.data
    bp = 0

#model.decoder.prelu.weight.requires_grad = False

#introduce a couple of fixed point parameters
fp_bits = 3         # number of bits used to represent the weights
fp_range = 2**fp_bits      # the max value

# the min/max number (-fp_range/2 <= x < fp_range/2)
#fp_min = -(fp_range >> 1)
#fp_max = (fp_range >> 1) - 1
#scale = 13.2 * (fp_range >> 1)    # this is the scale factor to divide the final number by

# the min/max number (0 <= x < fp_range)
fp_min = 0
fp_max = fp_range - 1
scale = 7     # this is the scale factor to divide the final number by


# use something like this to manually set the weights.  use the no_grad() to prevent tracking of gradient changes
with torch.no_grad():
    rnd_range = 1/fp_range
    mr = np.random.default_rng(10)

    input_layer_shape = model.decoder.input_layer.weight.data.shape
    output_layer_shape = model.decoder.output_layer.weight.data.shape

    # normal random numbers
    #model.decoder.input_layer.weight.data = nn.Parameter(torch.from_numpy(mr.integers(fp_min, fp_max, input_layer_shape).astype(np.float32)/(scale))).to(device)
    #model.decoder.output_layer.weight.data = nn.Parameter(torch.from_numpy(mr.integers(fp_min, fp_max, output_layer_shape).astype(np.float32)/(scale))).to(device)
    #model.decoder.input_layer.weight.data = nn.Parameter(torch.from_numpy(mr.uniform(-rnd_range, rnd_range, input_layer_shape).astype(np.float32))).to(device)
    #model.decoder.output_layer.weight.data = nn.Parameter(torch.from_numpy(mr.uniform(-rnd_range, rnd_range, output_layer_shape).astype(np.float32))).to(device)
    # model.decoder.input_layer.weight.data = nn.Parameter(torch.from_numpy((fp_max >> 1) * np.ones(input_layer_shape).astype(np.float32))).to(device)
    # model.decoder.output_layer.weight.data = nn.Parameter(torch.from_numpy((fp_max >> 1) * np.ones(output_layer_shape).astype(np.float32))).to(device)
    model.decoder.input_layer.weight.data = nn.Parameter(torch.from_numpy(np.ones(input_layer_shape).astype(np.float32))).to(device)
    model.decoder.output_layer.weight.data = nn.Parameter(torch.from_numpy(np.ones(output_layer_shape).astype(np.float32))).to(device)
    #np.ones((1, 1, 1, feature_size)

    # make a deep copy of the weights to make sure they don't change
    # dw1a = copy.deepcopy(model.decoder.input_layer.weight)
    # dw2a = copy.deepcopy(model.decoder.output_layer.weight)


# this is setup as a static learning rate.  we may want to look at variable lr based on some performance numbers
#optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-1, div_factor=100, steps_per_epoch=1, epochs=max_epochs, verbose=True)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5000, threshold=0.1, threshold_mode='rel', cooldown=20, min_lr=1e-10, eps=1e-08, verbose=True)
criterion = nn.MSELoss()
#criterion = nn.L1Loss()

'''
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="D:/data", train=True, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
)
'''

#cv2.setNumThreads(0)

#torch.multiprocessing.set_start_method('spawn')
if __name__ == '__main__':

    idx = 0
    rng = np.random.default_rng(10)
    data_bits = 4
    data_min = 0
    data_max = 2**data_bits

    if(read_data == True):
        x = np.fromfile("../data/lfm_test_10M_100m_0000.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)
        x = x[np.s_[idx:(idx + io_size)]]
        data_type = "real"
    else:
        # normal range of IQ values
        #x = rng.integers(-2048, 2048, size=(1, 1, 1, io_size), dtype=np.int16, endpoint=False).astype(np.float32)
        # normal range of IQ values converted to unsigned with a shift
        #x = rng.integers(0, 4096, size=(1, 1, 1, io_size), dtype=np.int16, endpoint=False).astype(np.float32)
        # normal IQ values decomposed into 8-bit unsigned values
        x = rng.integers(data_min, data_max, size=(1, 1, 1, io_size), dtype=np.int16, endpoint=False).astype(np.float32)
        data_type = "{:02d}bit-uint".format(data_bits)

    # get the mean of x
    x_mean = math.floor(np.mean(x))

    # input into the decoder
    F = torch.from_numpy(data_max*0.5*np.ones((1, 1, 1, feature_size)).astype(np.float32)).to(device)
    F = F.view(-1, feature_size)

    # convert x into a torch tensor variable
    X = torch.from_numpy(x).to(device)
    X = X.view(-1, io_size)

    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_name = "fs{:02d}-io{:03d}-".format(feature_size, io_size) + data_type
    log_dir = "../results/" + scenario_name + "/"

    os.makedirs(log_dir, exist_ok=True)

    data_writer = open((log_dir + scenario_name + "_" + date_time + ".txt"), "w")

    # model must be set to train mode for QAT logic to work
    model.train()

    lr_shift = 10.0

    epoch_inc = 2000000

    data_writer.write("#-------------------------------------------------------------------------------\n")

    data_writer.write("# data bits, min, max:\n{}, {}, {}\n\n".format(data_bits, data_min, data_max))

    data_writer.write("# io_size:\n{}\n\n".format(io_size))
    data_writer.write("# feature_size:\n{}\n\n".format(feature_size))
    data_writer.write("# F:\n")

    for idx in range(feature_size):
        data_writer.write("{:.6f}".format((F.numpy())[0][idx]))
        if(idx<feature_size-1):
            data_writer.write(", ")
        else:
            data_writer.write("\n\n")

    # start the training
    for epoch in range(max_epochs):
        #model.train()
        loss = 0
        optimizer.zero_grad()
        outputs = model(F)

        train_loss = criterion(outputs, X)
        loss += train_loss.item()

        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, loss))

        loss_q = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
        if (loss_q < 1):
            bp = 10
            break

        train_loss.backward()
        optimizer.step()

        # different way of gradually reducing learning rate
        # if(loss < lr_shift):
        #     lr = optimizer.param_groups[0]['lr']
        #     optimizer.param_groups[0]['lr'] = 0.95*lr
        #     lr_shift = 0.9*lr_shift


    loss = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
    print("\nloss = {:.6f}".format(loss.item()))

    data_writer.write("#-------------------------------------------------------------------------------\n")
    data_writer.write("# final_loss:\n{:.4f}\n\n".format(loss.item()))
    data_writer.write("# X:\n")

    for idx in range(io_size):
        data_writer.write("{}".format((X.numpy())[0][idx]))
        if(idx<io_size-1):
            data_writer.write(", ")
        else:
            data_writer.write("\n")

    data_writer.close()

    model.eval()

    for fp_bits in range(math.floor(data_bits / 2)-1, data_bits + 1):

        fp_range = 2**fp_bits      # the max value

        # the min/max number (0 <= x < fp_range)
        fp_min = 0
        fp_max = fp_range - 1

        scale_step = 0.005

        min_scale = math.floor(fp_range/16.0)
        max_scale = (fp_range * 0.625) + scale_step
        min_loss = 1e10

        data_wr = open((log_dir + scenario_name + "_{:02d}-bits_".format(fp_bits) + date_time + ".txt"), "w")

        for scale in np.arange(min_scale, max_scale, scale_step):

            with torch.no_grad():
                #outputs = model(X)
                #outputs = torch.floor(outputs + 0.5)


                #dw1b = copy.deepcopy(model.decoder.input_layer.weight)
                dw2b = copy.deepcopy(model.decoder.output_layer.weight)
                #dw1b_q = torch.clamp_min(torch.clamp_max(torch.floor(dw1b * scale), fp_max), fp_min)
                dw2b_q = torch.clamp_min(torch.clamp_max(torch.floor(dw2b * scale), fp_max), fp_min)
                #d1 = torch.floor(dw1b_q + 0.5)/(scale)
                d2 = torch.floor(dw2b_q + 0.5)/(scale)

                #print("\nOriginal Input:\n", X)
                #print("\nOutput:\n",torch.floor(outputs + 0.5))

                Y = torch.floor(model.decoder(F) + 0.5)

                D = copy.deepcopy(model.decoder)
                #D.input_layer.weight.data = d1
                D.output_layer.weight.data = d2
                Y2 = torch.floor(D(F) + 0.5)
                loss2 = torch.sum(torch.abs(Y2 - X))
                # print("loss2 = {:.6f}".format(loss2.item()))
                print("scale = {:0.3f}, loss = {:.2f}".format(scale, loss2.item()))

                # if(loss2.item() < min_loss):
                #     min_loss = loss2.item()
                #     min_scale = scale

                # writer.add_scalar("Loss/scale - {:01d} bits ".format(fp_bits), loss2, scale*1000)

                data_wr.write("{:0.3f}, {}, ".format(scale, loss2))

                for idx in range(io_size):
                    data_wr.write("{}".format((Y2.numpy())[0][idx]))
                    if(idx<io_size-1):
                        data_wr.write(", ")
                    else:
                        data_wr.write("\n")

        data_wr.close()

        # writer.add_hparams({"epoch_inc": epoch_inc, "io_size": io_size, "feature_size": feature_size, "min_scale": min_scale,
        #                     "fp_bits": fp_bits, "fp_min": fp_min, "fp_max": fp_max},
        #                    {"FP Loss": min_loss})

    # close the writer
    # writer.flush()
    # writer.close()


    # just a stopping break point before the code ends
    bp = 9