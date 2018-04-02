# -*- coding: utf-8 -*-
__author__ = 'huangyf'

import os
import sys
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
from utils import AverageMeter, accuracy, ImageTransformations, LabelTransformations, Logger

use_gpu = torch.cuda.is_available()


def train_model(model, criterion, optimizer, scheduler, log_saver, mode, num_epochs=60):
    since = time.time()
    steps = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        loss_meter_test = AverageMeter()
        acc_meter_test = AverageMeter()

        for phase in ['train', 'test']:

            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            for i, data in enumerate(loaders[phase]):
                inputs, labels = data
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                if phase == 'train':
                    inputs = Variable(inputs)
                    labels = Variable(labels)
                else:
                    inputs = Variable(inputs, volatile=True)
                    labels = Variable(labels, volatile=True)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                    loss_meter.update(loss.data[0], outputs.size(0))
                    acc_meter.update(accuracy(outputs.data, labels.data)[-1][0], outputs.size(0))
                    steps += 1
                    log_saver.log_train(steps, loss_meter.avg, acc_meter.avg)

                else:
                    loss_meter_test.update(loss.data[0], outputs.size(0))
                    acc_meter_test.update(accuracy(outputs.data, labels.data)[-1][0], outputs.size(0))

            if phase == 'train':
                epoch_loss = loss_meter.avg
                epoch_acc = acc_meter.avg

            else:
                epoch_loss = loss_meter_test.avg
                epoch_acc = acc_meter_test.avg
                log_saver.log_test(steps, loss_meter_test.avg, acc_meter_test.avg)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        print()

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print('Saving..')
            state = {
                'net': model,
                'epoch': epoch,
                'loss': loss_meter.avg,
                'acc': acc_meter.avg,
                'loss_test': loss_meter_test.avg,
                'acc_test': acc_meter_test.avg,
                'log': log_saver
            }
            if not os.path.isdir('checkpoint_{}'.format(mode[2])):
                os.mkdir('checkpoint_{}'.format(mode[2]))
            torch.save(state, './checkpoint_{}/{}_{}_ckpt_epoch_{}.t7'.format(mode[2], mode[0], mode[1], epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model, log_saver


root = './'
BATCH_SIZE = 128
weight_decay = 0.
num_epochs = 60

for mode1 in ['normal', 'random', 'shuffled']:
    for mode2 in ['normal', 'random', 'partially']:
        img_transforms = ImageTransformations(mode=mode1)
        label_transforms = LabelTransformations(mode=mode2)
        training_dataset = CIFAR10(root, train=True, transform=img_transforms, target_transform=label_transforms)
        training_loader = DataLoader(training_dataset, BATCH_SIZE, shuffle=True, pin_memory=True)

        testing_dataset = CIFAR10(root, train=False,
                                  transform=transforms.Compose([Image.fromarray, transforms.ToTensor(),
                                                                transforms.Normalize(
                                                                    (0.4914, 0.4822, 0.4465),
                                                                    (0.2023, 0.1994, 0.2010))]))
        testing_loader = DataLoader(testing_dataset, BATCH_SIZE, shuffle=False, pin_memory=True)

        loaders = {'train': training_loader, 'test': testing_loader}

        resnet18 = resnet.ResNet18()
        vgg16 = vgg.VGG('VGG16')
        alex = alexnet.alexnet()
        inception = inceptions.GoogLeNet()

        model = resnet18
        if use_gpu:
            model = resnet18.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=weight_decay)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        mode = [mode1, mode2, 'resnet']
        log = Logger(mode)

        model, log = train_model(model, criterion, optimizer, exp_lr_scheduler, log, mode, num_epochs=num_epochs)


def plot(title):
    checkpoints = os.listdir('./checkpoint_{}'.format(title))
    fig = plt.figure(1, figsize=(20, 10))
    fig.suptitle(title)

    ax_train_loss = fig.add_subplot(221)
    ax_test_loss = fig.add_subplot(222)
    ax_train_acc = fig.add_subplot(223)
    ax_test_acc = fig.add_subplot(224)
    lines = []
    labels = []
    for checkpoint in checkpoints:
        if int(checkpoint.split('_')[-1].split('.')[0]) != num_epochs - 1:
            continue
        mode = checkpoint.split('_')[:2]
        state = torch.load(os.path.join('./checkpoint_{}'.format(title), checkpoint))
        # num_params=sum(p.numel() for p in state['net'].parameters() if p.requires_grad)
        log = state['log']
        labels.append(mode[0] + '-' + mode[1])
        line = ax_train_loss.plot(log.step_logger, log.loss_logger)
        lines.append(line[0])
        ax_train_acc.plot(log.step_logger, log.acc_logger)
        ax_test_loss.plot(log.step_logger_test, log.loss_logger_test)
        ax_test_acc.plot(log.step_logger_test, log.acc_logger_test)
    for ax in [ax_train_loss, ax_train_acc, ax_test_loss, ax_test_acc]:
        ax.set_xlim(0, len(log.step_logger))
        ax.set_xlabel('steps')

    ax_train_loss.set_ylabel('train_loss')
    ax_train_acc.set_ylabel('train acc')
    ax_test_loss.set_ylabel('test loss')
    ax_test_acc.set_ylabel('test acc')

    fig.legend(lines, labels, loc='upper left')
    plt.savefig('comparision.png')


plot('resnet')
