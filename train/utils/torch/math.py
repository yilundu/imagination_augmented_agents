import torch


def standard_normal(*args):
    return torch.normal(torch.zeros(*args), torch.ones(*args)).cuda()
