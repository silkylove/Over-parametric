# -*- coding: utf-8 -*-
__author__ = 'huangyf'

import os
import pickle
import torch
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from cifar import CIFAR10
from models import resnet, alexnet, inceptions, vgg
from utils import AverageMeter, accuracy, ImageTransformations, LabelTransformations

use_gpu = torch.cuda.is_available()


def train_model(model, criterion, optimizer, scheduler, num_epochs=60):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        scheduler.step()
        model.train(True)

        for data in training_loader:
            inputs, labels = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_meter.update(loss.data[0], BATCH_SIZE)
            acc_meter.update(accuracy(outputs.data, labels.data))

        epoch_loss = loss_meter.avg
        epoch_acc = acc_meter.avg

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'train', epoch_loss, epoch_acc))

        print()

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print('Saving..')
            state = {
                'net': model.module if use_gpu else model,
                'loss': epoch_loss,
                'acc': epoch_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_epoch_{}.t7'.format(epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model


root = './'
BATCH_SIZE = 100
weight_decay = 0.
img_transforms = ImageTransformations(mode='random')
label_transforms = LabelTransformations(mode='random')
training_dataset = CIFAR10(root, train=True, transform=img_transforms, target_transform=label_transforms)
training_loader = DataLoader(training_dataset, BATCH_SIZE, shuffle=True, pin_memory=True)

resnet18 = resnet.ResNet18()
vgg16 = vgg.VGG('VGG16')
alex = alexnet.alexnet()
inception = inceptions.GoogLeNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(training_dataset) // BATCH_SIZE, gamma=0.95)
