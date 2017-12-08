# reference: https://github.com/ahirner/pytorch-retraining/blob/master/retrain_benchmark_bees.ipynb
import logging
import os

import torch
import torchvision

thisfile = os.path.abspath(__file__)

model_path = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-241335ed.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-6f0f7f60.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-4c113574.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-17b70270.pth',
    'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth'
}


def load_state_dict(model, state_dict):
    thisfunc = thisfile + '->load_state_dict()'

    # replace unusable parameters
    params = []
    for param, x in model.state_dict().items():
        if param in state_dict:
            y = state_dict[param]
            if hasattr(x, 'size') and hasattr(y, 'size') and x.size() != y.size():
                params.append(param)
                state_dict[param] = x
        else:
            params.append(param)
            state_dict[param] = x

    if len(params) > 0:
        logging.warning('{0}: replacing params {1} by initial values'.format(thisfunc, params))

    # remove redundant parameters
    params = []
    for param in state_dict.keys():
        if param not in model.state_dict():
            params.append(param)

    for param in params:
        del state_dict[param]

    if len(params) > 0:
        logging.warning('{0}: removing params {1}'.format(thisfunc, params))

    # sanity check
    assert len([param for param in model.state_dict().keys() if param not in state_dict.keys()]) == 0
    assert len([param for param in state_dict.keys() if param not in model.state_dict().keys()]) == 0

    # load state dict
    model.load_state_dict(state_dict)


def load_imagenet_model(name, *args, **kwargs):
    # load model and state dict
    model = getattr(torchvision.models, name)(*args, **kwargs)
    state_dict = torch.utils.model_zoo.load_url(model_path[name])

    # load state dict to model
    load_state_dict(model, state_dict)
    return model
