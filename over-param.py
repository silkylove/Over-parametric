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
from models import resnet
from utils import AverageMeter,accuracy,ImageTransformations,LabelTransformations

use_gpu = torch.cuda.is_available()








def train_model(model, criterion, optimizer, scheduler, num_epochs=60):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        model.train(True)

        running_loss = 0.0
        running_corrects = 0

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

            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(training_dataset)
        epoch_acc = running_corrects / len(training_dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'train', epoch_loss, epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model


root = './'
BATCH_SIZE = 100
img_transforms = ImageTransformations(mode='random')
label_transforms = LabelTransformations(mode='random')
training_dataset = CIFAR10(root, train=True, transform=img_transforms, target_transform=label_transforms)
training_loader = DataLoader(training_dataset, BATCH_SIZE, shuffle=True, pin_memory=True)
