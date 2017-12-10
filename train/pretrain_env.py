from __future__ import print_function

import argparse
import os
from multiprocessing import Pool

import utils
import numpy as np
import torch
# from hparams import hp
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm
import torchvision.models as models
from tqdm import tqdm

from pretrain_env_dataloader import PretrainDataset
from networks import EnvModel
import time
import random
import scipy.misc


def reconstruct_image(im):
    im = im.numpy()
    im = np.transpose(im, (1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    im = 256 * (im * std + mean)
    return im


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.3 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


NAME_TO_MODEL = {
	'env-model': EnvModel()
}


if __name__ == '__main__':
    default_path = '../data'
    noise_decay = 0.55
    loss_fn = MSELoss()

    # set up argument parser
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument('--exp', default = 'first')
    parser.add_argument('--resume', default = None, type = str)
    parser.add_argument('--clean', action = 'store_true')

    # dataset
    parser.add_argument('--data_path', default = default_path)
    parser.add_argument('--synset', default = '')
    parser.add_argument('--categories', default = '../data/categories.txt')

    # training
    parser.add_argument('--epochs', default = 500, type = int)
    parser.add_argument('--batch', default = 16, type = int)
    parser.add_argument('--snapshot', default = 2, type = int)
    parser.add_argument('--workers', default = 8, type = int)
    parser.add_argument('--gpu', default = '7')
    parser.add_argument('--name', default = 'env-model')

    # Training Parameters
    parser.add_argument('--lr', default = 0.1, type = float)
    parser.add_argument('--momentum', default = 0.9, type = float)
    parser.add_argument('--weight_decay', default = 1e-5, type = float)
    parser.add_argument('--noise', default = 0, type = float)

    # parse arguments
    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    # set up gpus for training
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # set up datasets and loaders
    data, loaders = {}, {}
    for split in ['train']:
        # if split == 'train':
        data[split] = PretrainDataset(data_path = os.path.join(args.data_path, args.synset), split = split)
        # else:
        #     data[split] = PretrainDataset(data_path = os.path.join(args.data_path, args.synset), split = split, augment= False)
        loaders[split] = DataLoader(data[split], batch_size = args.batch, shuffle = True, num_workers = args.workers)
    print('==> dataset loaded')
    print('[size] = {0}'.format(len(data['train'])))
    # print('[size] = {0} + {1}'.format(len(data['train']), len(data['val'])))

    # set up map for different categories
    # categories = np.genfromtxt(args.categories, dtype='str')[:, 0]

    # set up model and convert into cuda
    model = NAME_TO_MODEL[args.name].cuda()
    print('==> model loaded')
    best_loss = 10000.0

    # set up optimizer for training
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum = args.momentum,
                                nesterov = True,
                                weight_decay = args.weight_decay)
    print('==> optimizer loaded')

    # set up experiment path
    exp_path = os.path.join('exp', args.exp)
    utils.shell.mkdir(exp_path, clean = args.clean)
    logger = utils.Logger(exp_path)
    print('==> save logs to {0}'.format(exp_path))

    # load snapshot of model and optimizer
    if args.resume is not None:
        if os.path.isfile(args.resume):
            snapshot = torch.load(args.resume)
            epoch = snapshot['epoch']
            model.load_state_dict(snapshot['model'])
            # If this doesn't work, can use optimizer.load_state_dict
            optimizer.load_state_dict(snapshot['optimizer'])
            print('==> snapshot "{0}" loaded (epoch {1})'.format(args.resume, epoch))
        else:
            raise FileNotFoundError('no snapshot found at "{0}"'.format(args.resume))
    else:
        epoch = 0

    for epoch in range(epoch, args.epochs):
        step = epoch * len(data['train'])
        adjust_learning_rate(optimizer, epoch)

        # training the model
        model.train()
        sigma = args.noise / ((1 + epoch) ** noise_decay)

        losses = []
        losses_2 = []
        losses_3 = []
        count = 0
        for images, labels in tqdm(loaders['train'], desc = 'epoch %d' % (epoch + 1)):
            count += 1
            if count > 20:
                continue
            # convert images and labels into cuda tensor
            images = Variable(images.cuda()).float()
            labels = Variable(labels.cuda())
            # initialize optimizer
            optimizer.zero_grad()

            # forward pass
            outputs = model.forward(images)
            # print("BLAH ", outputs.size(), labels.size())
            # print("BLEH ", images.size())
            # print(labels.size(), (images.size()))
            # print((images[:,3:4]).size())
            loss = loss_fn(outputs, labels.squeeze())
            losses.append(loss.data[0])
            loss_2 = loss_fn(images[:,0], labels.squeeze())
            losses_2.append(loss_2.data[0])
            loss_3 = loss_fn(Variable(torch.FloatTensor(np.zeros((16, 50, 50))).cuda()), labels.squeeze() - images[:,0])
            losses_3.append(loss_3.data[0])
            # add summary to logger
            logger.scalar_summary('loss', loss.data[0], step)
            step += args.batch

            # Add noise to gradients
            for w in model.parameters():
                if w.grad is not None:
                    w.grad.data.normal_(mean = 0, std = sigma)

            # backward pass
            loss.backward()

            # Clip gradient norms
            clip_grad_norm(model.parameters(), 10.0)

            optimizer.step()

        print("Training loss: ", np.array(losses).mean(), len(losses))
        print("Baseline loss: ", np.array(losses_2).mean(), " ", np.array(losses_3).mean())
        # if args.snapshot != 0 and (epoch + 1) % args.snapshot == 0:

        if np.array(losses).mean() < best_loss:
            best_loss = np.array(losses).mean()

            # snapshot model and optimizer
            snapshot = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(snapshot, os.path.join(exp_path, 'best.pth'))
            print('==> saved snapshot to "{0}"'.format(os.path.join(exp_path, 'best.pth')))

        # Visualize
        # rand_index = random.randint(0, len(data['train']))
        model.eval()
        img = data['train'][0][0].unsqueeze_(0)
        for time_step in range(10):
            # print(img.size())
            images = Variable(img.cuda()).float()
            outputs = model.forward(images).data[0][0].cpu()

            next_frame = data['train'][time_step + 1][0]
            # print((img[0][1], img[0][2], outputs, next_frame[3]))
            img = torch.stack((img[0][1], img[0][2], outputs, next_frame[3]), 0).unsqueeze_(0)

            baseline_frame = data['train'][time_step][1].numpy()
            # print(baseline_frame.shape)
            print(baseline_frame)
            scipy.misc.toimage(baseline_frame).save('images/baseline' + str(epoch + 1) + '-' + str(time_step + 1) + '.png')

            scipy.misc.toimage(outputs.numpy()).save('images/output' + str(epoch + 1) + '-' + str(time_step + 1) + '.png')
            print("Wrote to ", 'images/output' + str(epoch + 1) + '-' + str(time_step + 1) + '.png')
