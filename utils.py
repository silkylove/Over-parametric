# -*- coding: utf-8 -*-
__author__ = 'huangyf'

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import transforms


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Logger(object):
    def __init__(self, name):
        self.reset()
        self.name = name

    def reset(self):
        self.epoch_logger = []
        self.loss_logger = []
        self.acc_logger = []
        self.loss_logger_test = []
        self.acc_logger_test = []

    def log_train(self, epoch, loss, acc):
        self.epoch_logger.append(epoch)
        self.loss_logger.append(loss)
        self.acc_logger.append(acc)

    def log_test(self, loss, acc):
        self.loss_logger_test.append(loss)
        self.acc_logger_test.append(acc)


class Transformations():

    def __init__(self, mode='normal', mean_var=None):
        '''
        :param mode: normal,random,shuffled,gaussian
        :param mean_var: only used for image and mode = gaussian
        '''
        self.mode = mode

        self.mean_var = mean_var

    def __call__(self, data):
        pass


class ImageTransformations(Transformations):

    def __call__(self, image):
        h, w, c = image.shape
        if self.mode == 'normal':
            image = image
        elif self.mode == 'shuffled':
            np.random.seed(0)
            image = np.random.permutation(image.reshape(-1, )).reshape(h, w, c)
        elif self.mode == 'random':
            image = np.random.permutation(image.reshape(-1, )).reshape(h, w, c)
        elif self.mode == 'gaussian':
            image = np.random.randn(h, w, c) * self.mean_var[1] + self.mean_var[0]

        image = Image.fromarray(image)

        image = transforms.ToTensor()(image)
        image = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))(image)
        return image


class LabelTransformations(Transformations):

    def __call__(self, label):
        if self.mode == 'normal':
            return label
        elif self.mode == 'random':
            label = np.random.randint(0, 10)
        elif self.mode == 'partially':
            p = 0.5
            if np.random.uniform(0, 1) > p:
                label = np.random.randint(0, 10)
        return label
