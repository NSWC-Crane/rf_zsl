
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import copy
import math
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime


###torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

max_epochs = 300000

# number of random samples to generate (should be a multiple of two for flattening an IQ pair)
input_size = 8
feature_size = 3
decoder_int1 = 128
m = 36

read_data = True

# create the encoder class
class Encoder(nn.Module):
    def __init__(self, input_size, feature_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, feature_size, bias=False)
        self.hidden_layer_1 = nn.Linear(16, 16, bias=False)
        self.hidden_layer_2 = nn.Linear(128, 128, bias=False)
        #self.hidden_layer_3 = nn.Linear(32, 128, bias=False)
        self.output_layer = nn.Linear(16, feature_size, bias=False)
        self.prelu = nn.PReLU(1, 0.25)
        self.silu = nn.SiLU()
        self.elu = nn.ELU()
        self.tanshrink = nn.Tanhshrink()

    def forward(self, activation):
        activation = self.input_layer(activation)
        #activation = self.prelu(activation)
        #activation = self.hidden_layer_1(activation)
        #activation = self.silu(activation)
        #activation = self.elu(activation)
        #activation = self.prelu(activation)
        #activation = self.tanshrink(activation)
        #activation = self.hidden_layer_2(activation)
        #activation = self.prelu(activation)
        #activation = self.hidden_layer_3(activation)
        #activation = self.prelu(activation)
        #activation = self.output_layer(activation)
        #activation = self.prelu(activation)
        #activation = self.silu(activation)
        #activation = self.elu(activation)
        #activation = self.prelu(activation)
        #activation = self.tanshrink(activation)
        return activation

# create the decoder class
class Decoder(nn.Module):
    def __init__(self, output_size, feature_size):
        super().__init__()
        self.input_layer = nn.Linear(feature_size, decoder_int1, bias=False)
        self.hidden_layer_1 = nn.Linear(decoder_int1, 256, bias=False)
        self.hidden_layer_2 = nn.Linear(256, 5412, bias=False)
        self.hidden_layer_3 = nn.Linear(512, 1024, bias=False)
        self.hidden_layer_4 = nn.Linear(1024, 2048, bias=False)
        self.output_layer = nn.Linear(2048, output_size, bias=False)
        #self.prelu = nn.PReLU(1, 0.25)
        # self.multp = nn.Parameter(torch.tensor([[128.0]]))
        #self.silu = nn.SiLU()
        #self.elu = nn.ELU()
        #self.tanshrink = nn.Tanhshrink()
        #self.tanh = nn.Tanh()
        #self.relu = nn.ReLU()

    def forward(self, activation):
        # self.round_weights()
        #activation = self.prelu(activation)
        activation = self.input_layer(activation)
        #activation = self.elu(activation)
        #activation = self.prelu(activation)
        #activation = self.relu(activation)
        # activation = activation*self.multp.expand_as(activation)
        activation = self.hidden_layer_1(activation)
        activation = self.hidden_layer_2(activation)
        activation = self.hidden_layer_3(activation)
        activation = self.hidden_layer_4(activation)
        #activation = self.silu(activation)
        activation = self.output_layer(activation)
        #activation = self.tanh(activation)
        #activation = self.prelu(activation)
        #activation = self.silu(activation)
        #activation = self.elu(activation)
        #activation = activation
        return activation

    def round_weights(self):
        with torch.no_grad():
            for param in model.decoder.parameters():
                # t1 = model.decoder.input_layer.weight.data
                # t1 = 2 * (t1 > 0).type(torch.float32) - 1
                # t1 = torch.floor(t1*m + 0.5)/m
                # t1 = torch.clamp_min(torch.clamp_max(m*t1, 16), -16)
                # t1 = torch.floor(t1+0.5)/m
                t1 = param.data
                t1 = 2 * (t1 > 0).type(torch.float32) - 1
                t1 = t1 / m
                param.data = t1

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

# setup everything
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model = AE(input_size, feature_size).to(device)

# cycle through the decoder side and do whatever needs doing
# for parameter in model.decoder.parameters():
#     #parameter.requires_grad = False        # uncomment to freeze the training of these layers
#     p = parameter.data
#     bp = 0

#model.decoder.prelu.weight.requires_grad = False

# use something like this to manually set the weights.  use the no_grad() to prevent tracking of gradient changes
with torch.no_grad():
    #rnd_range = 1/128
    mr = np.random.default_rng(10)

    # random values of -1/1
    model.decoder.input_layer.weight.data = nn.Parameter(torch.from_numpy(2*(mr.uniform(0, 1.0, [decoder_int1, feature_size]) > 0.5).astype(np.float32)-1)).to(device)
    model.decoder.hidden_layer_1.weight.data = nn.Parameter(torch.from_numpy(2 * (mr.uniform(0, 1.0, [256, decoder_int1]) > 0.5).astype(np.float32) - 1)).to(device)
    model.decoder.hidden_layer_2.weight.data = nn.Parameter(torch.from_numpy(2 * (mr.uniform(0, 1.0, [512, 256]) > 0.5).astype(np.float32) - 1)).to(device)
    model.decoder.hidden_layer_3.weight.data = nn.Parameter(torch.from_numpy(2 * (mr.uniform(0, 1.0, [1024, 512]) > 0.5).astype(np.float32) - 1)).to(device)
    model.decoder.hidden_layer_4.weight.data = nn.Parameter(torch.from_numpy(2 * (mr.uniform(0, 1.0, [2048, 1024]) > 0.5).astype(np.float32) - 1)).to(device)
    model.decoder.output_layer.weight.data = nn.Parameter(torch.from_numpy(2*(mr.uniform(0, 1.0, [input_size, 2048]) > 0.5).astype(np.float32)-1)).to(device)

    # make a deep copy of the weights to make sure they don't change
    ew1 = copy.deepcopy(model.encoder.input_layer.weight)
    dw1a = copy.deepcopy(model.decoder.input_layer.weight)
    dw2a = copy.deepcopy(model.decoder.output_layer.weight)


# this is setup as a static learning rate.  we may want to look at variable lr based on some performance numbers
#optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)
criterion = nn.MSELoss()

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

    Xc = np.copy(X)
    Xc = torch.from_numpy(Xc).to(device)
    Xc = Xc.view(-1, input_size)
    X /= 2000

    # model must be set to train mode for QAT logic to work
    model.train()

    lr_shift = 1.0
    update_weights = 100

    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(f'../runs/{tag}')

    for epoch in range(max_epochs):
        model.train()
        loss = 0
        optimizer.zero_grad()
        outputs = model(X)
        # outputs = torch.floor(outputs + 0.5)

        train_loss = criterion(outputs, X)
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

        if (epoch % update_weights) == 0:
            model.decoder.round_weights()
            outputs = model(X)
            loss = criterion(2000*outputs, Xc)

        writer.add_scalar("Loss/train", loss, epoch)
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, max_epochs, (loss)))

        if (torch.sum(torch.abs(torch.floor(outputs + 0.5) - X)) < 1):
            break

        if(loss < lr_shift):
            lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = 0.95*lr
            lr_shift = 0.9*lr_shift

        # Check the accuracy after each epoch
        #quantized_model = torch.quantization.convert(model.eval(), inplace=False)
        #quantized_model.eval()

        #scheduler.step(math.floor(loss))
        #scheduler.step()

    model.decoder.round_weights()

    with torch.no_grad():
        #outputs = model(X)
        #outputs = torch.floor(outputs + 0.5)
        loss = torch.sum(torch.abs(torch.floor(outputs + 0.5) - X))
        print("\nloss = {:.6f}".format(loss.item()))
        outputs = model(X)
        # loss = torch.sum(torch.abs(torch.floor((outputs*1000) + 0.5) - Xc))
        # print("\nloss3 = {:.6f}".format(loss.item()))

        ew2 = copy.deepcopy(model.encoder.input_layer.weight)
        dw1b = copy.deepcopy(model.decoder.input_layer.weight)
        dw2b = copy.deepcopy(model.decoder.output_layer.weight)
        d1a = dw1b*128
        d2a = dw2b*128
        d1 = torch.floor(d1a+0.5)/128
        d2 = torch.floor(d2a+0.5)/128

        bp = 5
        #print("\nOriginal Input:\n", X)
        #print("\nOutput:\n",torch.floor(outputs + 0.5))

        f = model.encoder(X)

        D = copy.deepcopy(model.decoder)
        D.input_layer.weight.data = d1
        D.output_layer.weight.data = d2
        Y = torch.floor(D(f) + 0.5)
        Y2 = torch.floor(model.decoder(f) + 0.5)
        loss2 = torch.sum(torch.abs(Y - X))
        print("loss2 = {:.6f}".format(loss2.item()))

    torch.save(model, f'../runs/{tag}/saved_model.pth')
    writer.add_hparams({'epochs': max_epochs, 'update_weights': update_weights, 'lr_shift': lr_shift,
                        'input_size': input_size, 'feature size': feature_size, 'decoder_init1': decoder_int1, 'scale': m},
                       {'hparam/loss': loss})
    writer.add_graph(model, X)
    writer.close()

    bp = 9
