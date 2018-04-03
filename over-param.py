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
from utils import AverageMeter, accuracy, Logger

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
lr = 0.1
BATCH_SIZE = 128
weight_decay = 0.
num_epochs = 60

## 'vgg16','resnet18','alex','inception'
model_name = 'vgg16'

mode1_set = ['normal', 'random', 'shuffled']
mode2_set = ['normal', 'random', 'partially']

for mode1 in mode1_set:
    for mode2 in mode2_set:
        print(mode1, ' and ', mode2, ' :')
        img_transforms = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(
                                                 (0.4914, 0.4822, 0.4465),
                                                 (0.2023, 0.1994, 0.2010))])
        training_dataset = CIFAR10(root, train=True, transform=img_transforms, image_mode=mode1, label_mode=mode2)
        training_loader = DataLoader(training_dataset, BATCH_SIZE, shuffle=True, pin_memory=True)

        testing_dataset = CIFAR10(root, train=False, transform=img_transforms)
        testing_loader = DataLoader(testing_dataset, BATCH_SIZE, shuffle=False, pin_memory=True)

        loaders = {'train': training_loader, 'test': testing_loader}

        resnet18 = resnet.ResNet18()
        vgg16 = vgg.VGG('VGG16')
        alex = alexnet.alexnet()
        inception = inceptions.GoogLeNet()

        exec('model={}'.format(model_name))
        if use_gpu:
            model = resnet18.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        mode = [mode1, mode2, model_name]
        log = Logger(mode)

        model, log = train_model(model, criterion, optimizer, exp_lr_scheduler, log, mode, num_epochs=num_epochs)


def plot(title):
    fontdict = {'size': 30}

    def get_fig(i):
        fig = plt.figure(i, figsize=(20, 10))
        ax = fig.add_subplot(111)
        plt.title(title, fontsize=40, y=1.04)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        return fig, ax

    fig1, ax1 = get_fig(1)
    fig2, ax2 = get_fig(2)
    fig3, ax3 = get_fig(3)
    fig4, ax4 = get_fig(4)

    for mode1 in mode1_set:
        for mode2 in mode2_set:
            state = torch.load('./checkpoint_{}/{}_{}_ckpt_epoch_{}.t7'.format(title, mode1, mode2, num_epochs - 1))
            log = state['log']
            label = mode1 + '-' + mode2
            ax1.plot(log.step_logger, log.loss_logger, linewidth=3, label=label)
            ax2.plot(log.step_logger, log.acc_logger, linewidth=3, label=label)
            ax3.plot(log.step_logger_test, log.loss_logger_test, linewidth=3, label=label)
            ax4.plot(log.step_logger_test, log.acc_logger_test, linewidth=3, label=label)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(0, len(log.step_logger))
        ax.set_xlabel('steps', fontdict=fontdict)
        ax.legend(loc='upper right', fontsize=20)

    ax1.set_ylabel('train loss', fontdict=fontdict)
    ax2.set_ylabel('train acc', fontdict=fontdict)
    ax3.set_ylabel('test loss', fontdict=fontdict)
    ax4.set_ylabel('test acc', fontdict=fontdict)

    fig1.savefig(title + '-train-loss.png')
    fig2.savefig(title + '-train-acc.png')
    fig3.savefig(title + '-test-loss.png')
    fig4.savefig(title + '-test-acc.png')


plot(model_name)
plt.show()
plt.close()
