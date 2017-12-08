import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import hp


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    # TODO Use whatever weight initialization is necessary
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        m.weight.data.normal_(1, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()


class EnvModel(nn.Module):
    """Network which given an input image frame, predicts the subsequent frame"""
    def __init__(self):
        self.conv1 = nn.Conv2d(num_channels, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.apply(weights_init)

    def forward(self, input):
        x = F.elu(self.conv1(input))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

    def sample(self, input, n):
        """Samples a trajectory from environment model of length n"""
        pass


class ModelFree(nn.Module):
    """Model free path for predicting the action and value"""
    def __init__(self, num_channels=9, actions=4):
        self.lstm_dim = 256
        self.pre_lstm_input = 32*3*3

        self.conv1 = nn.Conv2d(num_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # TODO compute the correct dimension for input 5 by 5
        self.lstm = nn.LSTM(self.pre_lstm_input, self.lstm_dim)

        self.critic_linear = nn.Linear(self.lstm_dim, 1)
        self.actor_linear = nn.Linear(self.lstm_dim, actions)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        # Set the forget gate bias to be 0
        # Note gate biases are i, f, g, o
        self.lstm.bias_ih.data[self.lstm_dim:2*self.lstm_dim].fill_(0.5)
        self.lstm.bias_hh.data[self.lstm_dim:2*self.lstm_dim].fill_(0.5)

        self.train()

    def forward(self, inputs):
        input, (hx, cx) = inputs
        x = F.elu(self.conv1(input))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, self.pre_lstm_input)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)


class I3A(nn.Module):
    def __init__(self, env_model):
        self.encoder_lstm = nn.LSTM(hp.conv_output_dim, hp.encoder_output_dim)
        self.model_free = ModelFree()
        self.env_model = env_model

        self.lstm = nn.LSTM(hp.joint_input_dim, hp.lstm_output_dim)
        self.critic_linear = nn.Linear(hp.lstm_output_dim, 1)
        self.actor_linear = nn.Linear(hp.lstm_output_dim, actions)

    def forward(self, input):
        model_free = self.model_free(input)
        traj_encodings = []

        for i in range(self.traj_num):
