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

        action_tensor = torch.zeros(5, 50, 50).float()
        action_tensor[image[3][0][0]].fill_(1)

        image = torch.from_numpy(image[:3, :, :]).float()
        aug_image = torch.cat([image, action_tensor], 0)

        label = self.labels[index].astype(np.float32)
        label_tensor = self.preprocess(torch.from_numpy(label))

        return aug_image, label_tensor

    def __len__(self):
        return self.images.shape[0]#self.dataset['images'].shape[0]
