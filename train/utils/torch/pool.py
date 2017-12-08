import random

import torch
from torch.autograd import Variable


# reference: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/image_pool.py
class Pool:
    def __init__(self, capacity):
        self.capacity = capacity
        if self.capacity > 0:
            self.size = 0
            self.elements = []

    def query(self, elements):
        if self.capacity == 0:
            return elements
        choices = []
        for element in elements.data:
            element = torch.unsqueeze(element, 0)
            if self.size < self.capacity:
                self.size += 1
                self.elements.append(element)
                choices.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    index = random.randint(0, self.capacity - 1)
                    candidate = self.elements[index].clone()
                    self.elements[index] = element
                    choices.append(candidate)
                else:
                    choices.append(element)
        choices = Variable(torch.cat(choices, 0))
        return choices
