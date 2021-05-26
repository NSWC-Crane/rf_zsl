
import torch
import torch.nn as nn


# create the encoder class
class Encoder(nn.Module):
    def __init__(self, input_size, feature_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 16, bias=False)
        self.hidden_layer_1 = nn.Linear(16, 128, bias=False)
        self.hidden_layer_2 = nn.Linear(128, 256, bias=False)
        # self.hidden_layer_3 = nn.Linear(256, 512, bias=False)
        self.output_layer = nn.Linear(256, feature_size, bias=False)

    def forward(self, activation):
        activation = self.input_layer(activation)
        activation = self.hidden_layer_1(activation)
        activation = self.hidden_layer_2(activation)
        # activation = self.hidden_layer_3(activation)
        activation = self.output_layer(activation)
        return activation


# create the decoder class
class Decoder(nn.Module):
    def __init__(self, output_size, feature_size, decoder_int1):
        super().__init__()
        self.input_layer = nn.Linear(feature_size, decoder_int1, bias=False)
        # self.hidden_layer_1 = nn.Linear(decoder_int1, 128, bias=False)
        # self.hidden_layer_2 = nn.Linear(128, 128, bias=False)
        # self.hidden_layer_3 = nn.Linear(512, 1024, bias=False)
        # self.hidden_layer_4 = nn.Linear(1024, 2048, bias=False)
        self.output_layer = nn.Linear(decoder_int1, output_size, bias=False)

    def forward(self, activation):
        activation = self.input_layer(activation)
        # activation = self.hidden_layer_1(activation)
        # activation = self.hidden_layer_2(activation)
        # activation = self.hidden_layer_3(activation)
        # activation = self.hidden_layer_4(activation)
        activation = self.output_layer(activation)
        return activation


# use the encoder and decoder classes to build the autoencoder
class AE(nn.Module):
    def __init__(self, input_size, feature_size, decoder_int1):
        super().__init__()
        self.encoder = Encoder(input_size, feature_size)
        self.decoder = Decoder(input_size, feature_size, decoder_int1)

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

