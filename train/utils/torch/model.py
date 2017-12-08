import torch
import torch.nn as nn
from torch.autograd import Variable


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        m.weight.data.normal_(1, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()


class ConvEncoder2D(nn.Module):
    def __init__(self, num_layers, in_channels = 3, out_channels = 8, batch_norm = True):
        super(ConvEncoder2D, self).__init__()

        # set up number of layers
        if isinstance(num_layers, int):
            num_layers = [num_layers, 0]

        network = []

        # several 3x3 convolutional layers and max-pooling layers
        for k in range(num_layers[0]):
            # 2d convolution
            network.append(nn.Conv2d(in_channels, out_channels, 3, padding = 1))

            # batch normalization
            if batch_norm:
                network.append(nn.BatchNorm2d(out_channels))

            # non-linearity and max-pooling
            network.append(nn.LeakyReLU(0.2, True))
            network.append(nn.MaxPool2d(2))

            # double channel size
            in_channels = out_channels
            out_channels *= 2

        # several 1x1 convolutional layers
        for k in range(num_layers[1]):
            # 2d convolution
            network.append(nn.Conv2d(in_channels, in_channels, 1))

            # batch normalization
            if batch_norm:
                network.append(nn.BatchNorm2d(in_channels))

            # non-linearity
            network.append(nn.LeakyReLU(0.2, True))

        # set up modules for network
        self.network = nn.Sequential(*network)
        self.network.apply(weights_init)

    def forward(self, inputs):
        return self.network.forward(inputs)


class ConvEncoder3D(nn.Module):
    def __init__(self, num_layers, in_channels = 3, out_channels = 8, batch_norm = True):
        super(ConvEncoder3D, self).__init__()

        # set up number of layers
        if isinstance(num_layers, int):
            num_layers = [num_layers, 0]

        network = []

        # several 3x3 convolutional layers and max-pooling layers
        for k in range(num_layers[0]):
            # 3d convolution
            network.append(nn.Conv3d(in_channels, out_channels, 3, padding = 1))

            # batch normalization
            if batch_norm:
                network.append(nn.BatchNorm3d(out_channels))

            # non-linearity and max-pooling
            network.append(nn.LeakyReLU(0.2, True))
            network.append(nn.MaxPool3d(2))

            # double channel size
            in_channels = out_channels
            out_channels *= 2

        # several 1x1 convolutional layers
        for k in range(num_layers[1]):
            # 3d convolution
            network.append(nn.Conv3d(in_channels, in_channels, 1))

            # batch normalization
            if batch_norm:
                network.append(nn.BatchNorm3d(in_channels))

            # non-linearity
            network.append(nn.LeakyReLU(0.2, True))

        # set up modules for network
        self.network = nn.Sequential(*network)
        self.network.apply(weights_init)

    def forward(self, inputs):
        return self.network.forward(inputs)


class SeqEncoder(nn.Module):
    def __init__(self, input_size, feature_size = 128, num_layers = 1, hidden_size = 256, dropout = 0.9):
        super(SeqEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # set up modules for recurrent neural networks
        self.rnn = nn.LSTM(input_size = input_size,
                           hidden_size = hidden_size,
                           num_layers = num_layers,
                           batch_first = True,
                           dropout = dropout,
                           bidirectional = True)
        self.rnn.apply(weights_init)

        # set up modules to compute features
        self.feature = nn.Linear(hidden_size * 2, feature_size)
        self.feature.apply(weights_init)

    def forward(self, inputs):
        # set up batch size
        batch_size = inputs.size(0)

        # compute hidden and cell
        hidden = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).cuda())
        cell = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).cuda())
        hidden_cell = (hidden, cell)

        # recurrent neural networks
        _, (hidden, cell) = self.rnn.forward(inputs, hidden_cell)

        # concatenate hidden states from both directions (fixme: do not support multiple layers)
        hidden_forward, hidden_backward = torch.split(hidden, 1, 0)
        hidden = torch.cat([hidden_forward.squeeze(0), hidden_backward.squeeze(0)], 1)

        # compute features by hidden states
        features = self.feature.forward(hidden)
        return features
