#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from utils.weight_perturbation import *

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        # print('DatasetSplit idx', idxs, self.idxs)

    def __len__(self):
        # print('len', len(self.idxs), self.idxs)
        return len(self.idxs)

    def __getitem__(self, item):
        # print('__getitem__', item, self.idxs[item])
        image, label = self.dataset[self.idxs[item]]
        return image, label

# class DatasetSplit2(Dataset):
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = list(idxs)
#         # print('DatasetSplit idx', idxs, self.idxs)

#     def __len__(self):
#         # print('len', len(self.idxs), self.idxs)
#         return len(self.idxs)

#     def __getitem__(self, item):
#         # print('__getitem__', item, self.idxs[item])
#         # image, label = self.dataset[self.idxs[item]]
#         for image, label in iter(self.dataset):
#             print("image:\t", image)
#             print("label:\t", label)
#         return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, 
                 loss_func = nn.CrossEntropyLoss(), lu_loader=None,
                 optimizer = None):
        self.args = args
        self.loss_func = loss_func
        self.selected_clients = []
        self.optimizer = optimizer
        # salt batch_size = 1
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        # 分組後只要讀進來就好，就不用再拆了。所以 batch_size=1。
        # https://pytorch.org/docs/stable/data.html
        # if args.model == 'medcnn' and args.dataset == 'medicalmnist':
        #     self.ldr_train = DataLoader(DatasetSplit2(dataset, idxs), batch_size=1)
        # else :
        #     self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=1)
        if idxs is None:
            if lu_loader is None :
                self.ldr_train = DataLoader(dataset, batch_size = 1)
            else :
                self.ldr_train = lu_loader
        else :
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = 1)
        # print(idxs)
        # print('ldr_train',len(self.ldr_train),self.ldr_train)

    def train(self, net):
        net.train()
        # train and update
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # if self.optimizer_op == 'sgd' :
        #     optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # elif self.optimizer_op == 'adam' :
        #     optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        # elif self.optimizer_op == 'wposgd' :
        #     optimizer = WPOptim(params=net.parameters(), base_optimizer=torch.optim.SGD, 
        #             lr=self.args.lr, alpha=0.05, momentum=0.9, weight_decay=1e-4)
        # else :
        #     optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # print('batch_idx', batch_idx)
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                if log_probs.shape[0] != labels.shape[0]:
                    raise ValueError("Number of outputs and labels don't match.")
                loss = self.loss_func(log_probs, labels)
                # loss = nn.BCEWithLogitsLoss()(log_probs.squeeze(1), labels.squeeze(1))
                # loss = nn.CrossEntropyLoss()(log_probs.squeeze(1), labels.squeeze(1))
                loss.backward()
                if hasattr(self.optimizer, 'generate_delta') and callable(self.optimizer.generate_delta):
                    self.optimizer.generate_delta(zero_grad=True)
                self.optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                            100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net, sum(epoch_loss) / len(epoch_loss)

