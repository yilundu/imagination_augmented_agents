import os
import h5py
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
# from torchvision import transforms
import transforms
import torch
import time


class PretrainDataset(Dataset):
    def __init__(self, data_path, split, augment=True, load_everything=True):
        self.count = 0
        file_path = os.path.join(data_path, 'pretrain_data.npz')
        self.dataset = np.load(file_path)

        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225])
        self.normalize = transforms.NormalizeTensor(mean=0.446, std=0.226)

        # transform = [
        #     transforms.Scale(256),
        #     transforms.RandomCrop(224),
        #     # transforms.RandomResizedCrop(224)
        #     ]

        transform = []
        # transform = [transforms.ToTensor()]

        transform += [
            self.normalize]

        self.preprocess = transforms.Compose(transform)

        self.split = split
        self.load_everything = load_everything
        if self.load_everything:
            self.images = self.dataset['arr_0']
            self.labels = self.dataset['arr_1']
            # self.images = np.array(self.dataset['images'])

    def __getitem__(self, index):
        self.count += 1

        if self.load_everything:
            image = self.images[index].astype(np.float32)
        else:
            image = self.dataset['images'][index]
        # print(image[:3].shape)
        # print(image.shape)
        # img_data = image[3]
        # img_tensor = self.preprocess(Image.fromarray(img_data, 'L'))
        # img_data = np.transpose(image[:3], (1, 2, 0))
        # img_data = image
        # img_tensor = self.preprocess(Image.fromarray(img_data))#   , 'RGB'))
        imgs = [self.preprocess(torch.FloatTensor(image[i])).unsqueeze_(0) for i in range(3)]
        # imgs = [self.preprocess(Image.fromarray(image[i], 'L')) for i in range(3)]
        #action_tensor = torch.FloatTensor(image[3:])
        action_tensor = [torch.FloatTensor(np.tile(1 if image[3][0][0] == i else 0, (50, 50))) for i in range(5)]
        # print(len(action_tensor))
        action_tensor = torch.stack(action_tensor)
        # print(action_tensor.size())
        label = self.labels[index].astype(np.float32)
        # print(label.shape)
        label_tensor = self.preprocess(torch.FloatTensor(label))
        # print(label_tensor)
        # label_tensor = self.preprocess(Image.fromarray(label, 'L'))
        # label_tensor = torch.FloatTensor(np.array([label]))#.astype(int))

        # print(img_tensor.size())
        # print(imgs[0].size())
        # print(action_tensor.size())
        # print(label_tensor.size())
        return torch.cat((imgs[0], imgs[1], imgs[2], action_tensor), 0), label_tensor

    def __len__(self):
        return self.images.shape[0]#self.dataset['images'].shape[0]