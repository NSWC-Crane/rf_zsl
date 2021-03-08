'''
This file is an experiment to test compression using a decoder only with fixed point and or integer based weights
'''

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
io_size = 32
feature_size = 21
decoder_int1 = 21

read_data = False

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
fp_bits = 4         # number of bits used to represent the weights
fp_range = 2**fp_bits      # the max value

# the min/max number (-fp_range/2 <= x < fp_range/2)
#fp_min = -(fp_range >> 1)
#fp_max = (fp_range >> 1) - 1
#scale = 13.2 * (fp_range >> 1)    # this is the scale factor to divide the final number by

# the min/max number (0 <= x < fp_range)
fp_min = 0
fp_max = fp_range - 1
scale = 1     # this is the scale factor to divide the final number by


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
    model.decoder.input_layer.weight.data = nn.Parameter(torch.from_numpy((fp_max >> 1) * np.ones(input_layer_shape).astype(np.float32))).to(device)
    model.decoder.output_layer.weight.data = nn.Parameter(torch.from_numpy((fp_max >> 1) * np.ones(output_layer_shape).astype(np.float32))).to(device)
    #np.ones((1, 1, 1, feature_size)

    # make a deep copy of the weights to make sure they don't change
    dw1a = copy.deepcopy(model.decoder.input_layer.weight)
    dw2a = copy.deepcopy(model.decoder.output_layer.weight)


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
    rng = np.random.default_rng()

    if(read_data == True):
        x = np.fromfile("../data/lfm_test_10M_100m_0000.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)
        x = x[np.s_[idx:(idx + io_size)]]
    else:
        # normal range of IQ values
        #x = rng.integers(-2048, 2048, size=(1, 1, 1, io_size), dtype=np.int16, endpoint=False).astype(np.float32)
        # normal range of IQ values converted to unsigned with a shift
        #x = rng.integers(0, 4096, size=(1, 1, 1, io_size), dtype=np.int16, endpoint=False).astype(np.float32)
        # normal IQ values decomposed into 8-bit unsigned values
        x = rng.integers(0, 256, size=(1, 1, 1, io_size), dtype=np.int16, endpoint=False).astype(np.float32)

    # input into the decoder
    F = torch.from_numpy(np.ones((1, 1, 1, feature_size)).astype(np.float32)).to(device)
    F = F.view(-1, feature_size)

    # convert x into a torch tensor variable
    X = torch.from_numpy(x).to(device)
    X = X.view(-1, io_size)

    # set up the stuff for writing
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=("../results/fp_dec/"+date_time))

    # model must be set to train mode for QAT logic to work
    model.train()

    m = 128
    lr_shift = 10.0

    epoch_inc = 2000

    #writer.add_hparams({"epoch_inc": epoch_inc, "io_size": io_size, "feature_size": feature_size}, {"none": 0})
    writer.add_scalar("epoch_inc", epoch_inc)

    for epoch in range(max_epochs):
        #model.train()
        loss = 0
        optimizer.zero_grad()
        outputs = model(F)

        train_loss = criterion(outputs, X)
        loss += train_loss.item()

        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, (loss)))

        if (torch.sum(torch.abs(torch.floor(outputs + 0.5) - X)) < 1):
            bp = 10
            break

        # clamp the weights to a fixed point value
        if(epoch % epoch_inc == 0):
            with torch.no_grad():
                t1 = model.decoder.input_layer.weight.data
                t1a = torch.clamp_min(torch.clamp_max(t1 * scale, fp_max), fp_min)
                t1a = torch.floor(t1a+0.5)/(scale)
                model.decoder.input_layer.weight.data = t1a

                t2 = model.decoder.output_layer.weight.data
                t2a = torch.clamp_min(torch.clamp_max(t2 * scale, fp_max), fp_min)
                t2a = torch.floor(t2a+0.5)/(scale)
                model.decoder.output_layer.weight.data = t2a

        # save the results
        writer.add_scalar("Loss/train", loss, epoch)

        train_loss.backward()
        optimizer.step()

        # different way of gradually reducing learning rate
        if(loss < lr_shift):
            lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = 0.95*lr
            lr_shift = 0.9*lr_shift

        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        # play with loss schedule
        #scheduler.step(math.floor(loss))
        #scheduler.step()

    model.eval()
    with torch.no_grad():
        #outputs = model(X)
        #outputs = torch.floor(outputs + 0.5)
        loss = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
        print("\nloss = {:.6f}".format(loss.item()))

        dw1b = copy.deepcopy(model.decoder.input_layer.weight)
        dw2b = copy.deepcopy(model.decoder.output_layer.weight)
        dw1b_q = torch.clamp_min(torch.clamp_max(dw1b * scale, fp_max), fp_min)
        dw2b_q = torch.clamp_min(torch.clamp_max(dw2b * scale, fp_max), fp_min)
        d1 = torch.floor(dw1b_q + 0.5)/(scale)
        d2 = torch.floor(dw2b_q + 0.5)/(scale)

        #print("\nOriginal Input:\n", X)
        #print("\nOutput:\n",torch.floor(outputs + 0.5))

        Y = torch.floor(model.decoder(F) + 0.5)

        D = copy.deepcopy(model.decoder)
        D.input_layer.weight.data = d1
        D.output_layer.weight.data = d2
        Y2 = torch.floor(D(F) + 0.5)
        loss2 = torch.sum(torch.abs(Y2 - X))
        print("loss2 = {:.6f}".format(loss2.item()))

    # close the writer
    writer.flush()
    writer.close()

    # just a stopping break point before the code ends
    bp = 9
