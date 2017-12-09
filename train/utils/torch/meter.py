import numpy as np
import torch


def to_numpy(x):
    return x.data.cpu().squeeze().numpy()


# reference: https://github.com/pytorch/tnt/blob/master/torchnet/meter/classerrormeter.py
class ClassErrorMeter:
    def __init__(self, topk = [1]):
        self.topk = np.sort(topk)
        self.reset()

    def reset(self):
        self.size = 0
        self.errors = {k: 0 for k in self.topk}

    def add(self, outputs, targets):
        outputs = to_numpy(outputs)
        targets = to_numpy(targets)

        if np.ndim(targets) == 2:
            targets = np.argmax(targets, 1)

        assert np.ndim(outputs) == 2, 'wrong output size (1D or 2D expected)'
        assert np.ndim(targets) == 1, 'wrong target size (1D or 2D expected)'
        assert targets.shape[0] == outputs.shape[0], 'number of outputs and targets do not match'

        topk = self.topk
        maxk = int(topk[-1])
        size = outputs.shape[0]

        predicts = torch.from_numpy(outputs).topk(maxk, 1, True, True)[1].numpy()
        corrects = (predicts == targets[:, np.newaxis].repeat(predicts.shape[1], 1))

        self.size += size
        for k in topk:
            self.errors[k] += size - corrects[:, :k].sum()

    def value(self, k = None, accuracy = True):
        assert k is None or k in self.topk, 'invalid k (this k was not provided at construction time)'

        if k is not None:
            value = float(self.errors[k]) / self.size * 100.
            if accuracy:
                return 100. - value
            else:
                return value
        else:
            values = [self.value(k, accuracy) for k in self.topk]
            if len(values) == 1:
                return values[0]
            else:
                return values


# reference: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
class ConfusionMeter:
    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.confusion = np.ndarray((nclasses, nclasses), dtype = np.int32)
        self.reset()

    def reset(self):
        self.confusion.fill(0)

    def add(self, outputs, targets):
        outputs = to_numpy(outputs)
        targets = to_numpy(targets)

        if np.ndim(targets) == 2:
            targets = np.argmax(targets, 1)

        assert outputs.shape[0] == targets.shape[0], 'number of targets and outputs do not match'
        assert outputs.shape[1] == self.nclasses, 'number of outputs does not match size of confusion matrix'

        predicts = outputs.argmax(1)
        for k, predict in enumerate(predicts):
            target = int(targets[k])
            self.confusion[target][predict] += 1

    def value(self, normalize = True):
        if normalize:
            confusion = self.confusion.astype(np.float32)
            return confusion / confusion.sum(1).clip(min = 1e-12)[:, None]
        else:
            return self.confusion
