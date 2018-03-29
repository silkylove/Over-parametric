# -*- coding: utf-8 -*-
__author__ = 'huangyf'

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

    def log(self, epoch, loss, acc):
        self.epoch_logger.append(epoch)
        self.loss_logger.append(loss)
        self.acc_logger.append(acc)


class Transformations():

    def __init__(self, mode='random'):
        self.mode = mode

    def __call__(self, data):
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


class ImageTransformations(Transformations):

    def __call__(self, image):
        pass


class LabelTransformations(Transformations):

    def __call__(self, label):
        pass
