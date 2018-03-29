# -*- coding: utf-8 -*-
__author__ = 'huangyf'
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


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


class Transformations():

    def __init__(self, mode='random'):
        self.mode = mode

    def __call__(self, data):
        pass


class ImageTransformations(Transformations):

    def __call__(self, image):
        pass


class LabelTransformations(Transformations):

    def __call__(self, label):
        pass
