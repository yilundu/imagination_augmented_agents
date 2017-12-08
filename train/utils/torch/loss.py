import torch
import torch.nn as nn
from torch.autograd import Variable


# reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class GANLoss(nn.Module):
    def __init__(self, real_label = 1.0, fake_label = 0.0, use_lsgan = True):
        super(GANLoss, self).__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.real_target = None
        self.fake_target = None
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def forward(self, output, target_is_real):
        if target_is_real:
            if self.real_target is None or self.real_target.numel() != output.numel():
                real_tensor = torch.Tensor(output.size()).fill_(self.real_label)
                self.real_target = Variable(real_tensor.cuda(), requires_grad = False)
            target = self.real_target
        else:
            if self.fake_target is None or self.fake_target.numel() != output.numel():
                fake_tensor = torch.Tensor(output.size()).fill_(self.fake_label)
                self.fake_target = Variable(fake_tensor.cuda(), requires_grad = False)
            target = self.fake_target
        return self.loss(output, target)
