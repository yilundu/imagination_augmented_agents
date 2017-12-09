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
    """Network which given an input image frame consisting of last 3 frames and action
       , predicts the subsequent frame"""
    def __init__(self, num_channels=4):
        super(EnvModel, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 96, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
        self.conv_predict = nn.Conv2d(96, 1, 3, stride=1, padding=1)

        self.apply(weights_init)

    def forward(self, input):
        x = F.elu(self.conv1(input))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.conv_predict(x)

        return x


class ModelFree(nn.Module):
    """Model free path for predicting the action and value"""
    def __init__(self, num_channels=3, actions=4):
        super(ModelFree, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.actor_linear = nn.Linear(hp.model_conv_output_dim, actions)
        self.embed_linear = nn.Linear(hp.model_conv_output_dim, hp.model_output_dim)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

    def forward(self, input):
        x = F.elu(self.conv1(input))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, hp.model_conv_output_dim)
        return self.actor_linear(x), self.embed_linear(x)


class I3A(nn.Module):
    def __init__(self, env_model, actions=4):
        super(I3A, self).__init__()
        self.encoder_lstm = nn.LSTM(hp.conv_output_dim, hp.encoder_output_dim)
        self.model_free = ModelFree()
        self.env_model = env_model

        self.lstm = nn.LSTM(hp.joint_input_dim, hp.lstm_output_dim)
        self.critic_linear = nn.Linear(hp.lstm_output_dim, 1)
        self.actor_linear = nn.Linear(hp.lstm_output_dim, actions)

        # Encoder convolution network
        self.e_conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.e_conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.e_conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.e_conv4 = nn.Conv2d(32, 16, 3, stride=2, padding=1)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.encoder_lstm.bias_ih_l0.data.fill_(0)
        self.encoder_lstm.bias_hh_l0.data.fill_(0)

        # Set the forget gate bias to be 0
        # Note gate biases are i, f, g, o
        l = hp.lstm_output_dim
        e = hp.encoder_output_dim
        self.lstm.bias_ih_l0.data[l:2*l].fill_(0.5)
        self.lstm.bias_hh_l0.data[l:2*l].fill_(0.5)
        self.encoder_lstm.bias_ih_l0.data[e:2*e].fill_(0.5)
        self.encoder_lstm.bias_hh_l0.data[e:2*e].fill_(0.5)

        self.train()

    def forward(self, inputs):
        (input, (hx, cx)) = inputs
        traj_encodings = []

        for i in range(hp.traj_num):
            traj_encodings += [self.sample_env(input, hp.traj_length)]

        traj_encoding = torch.cat(traj_encodings, dim=1)
        m_free_log, model_free_encoding = self.model_free(input)

        combined_enc = torch.cat([traj_encoding, model_free_encoding], dim=1)

        hx, cx = self.lstm(combined_enc, (hx, cx))
        x = hx.view(-1, hp.lstm_output_dim)

        return self.critic_linear(x), self.actor_linear(x), m_free_log, (hx, cx)


    def sample_env(self, input, n):
        """Input should be nx3x50x50 tensor of the past 3 states observed"""
        frames = list(torch.split(input, 1, 1))
        batch_size = input.size(0)

        # First generate our list of observations
        for i in range(n):
            input = torch.cat(frames[i:i+3], dim=1)
            actions, _ = self.model_free(input)

            # For stability, detach the gradient calculated for actions
            actions = F.softmax(actions)
            actions = actions.multinomial().float()
            actions = actions.detach()

            action_conv = actions.view(batch_size, 1, 1, 1).contiguous()
            action_conv = action_conv.expand(batch_size, 1, 50, 50).contiguous()

            conv_input = torch.cat([input, action_conv], 1)

            o_frame = self.env_model(conv_input)
            frames.append(o_frame)

        # Remove the first 3 input frames
        frames = frames[3:]
        frames_rev = frames[::-1]

        # Each frame should now be nx1x50x50
        frame_seq = torch.cat(frames_rev, dim=1)
        # Now each frame is size n x traj_len x 50 x 50
        frame_seq = frame_seq.view(-1, 1, 50, 50)
        # Feed through convolutional networks
        x = F.elu(self.e_conv1(frame_seq))
        x = F.elu(self.e_conv2(x))
        x = F.elu(self.e_conv3(x))
        x = F.elu(self.e_conv4(x))

        frame_seq = x.view(batch_size, hp.traj_length, -1)
        frame_seq = frame_seq.transpose(0, 1)
        output, (hx, cx) = self.encoder_lstm(frame_seq)

        cx = cx.transpose(0, 1).contiguous()
        batch_size = cx.size(0)

        return cx.view(-1, hp.encoder_output_dim)
