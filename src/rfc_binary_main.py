
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import copy
import math
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
import datetime

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

    # model must be set to train mode for QAT logic to work
    model.train()
    lr_shift = 1.0

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
            model.round_weights()
            outputs = model(X)
            loss = criterion(outputs, X)

        # writer.add_scalar("Loss/train", loss, epoch)
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

    model.round_weights()

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

    # torch.save(model, f'../runs/{tag}/saved_model.pth')
    # writer.add_hparams({'epochs': max_epochs, 'update_weights': update_weights, 'lr_shift': lr_shift,
    #                     'input_size': input_size, 'feature size': feature_size, 'decoder_init1': decoder_int1, 'scale': m},
    #                    {'hparam/loss': loss})
    # writer.add_graph(model, X)
    # writer.close()

    bp = 9
