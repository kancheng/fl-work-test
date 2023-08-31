#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.10

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

def train_segmentation(model, device, trainloader, optimizer, loss_function):
    model.train()
    running_loss = 0
    iou = []
    for i, (input, mask) in enumerate(trainloader):
        # load data into cuda
        input, mask = input.to(device), mask.to(device)
        # forward
        predict = model(input)
        loss = loss_function(predict, mask)
        # metric
        iou.append(get_iou_score(predict, mask).mean())
        running_loss += (loss.item())
        # zero the gradient + backprpagation + step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log the first image of the batch
        if ((i + 1) % 10) == 0:
            pred = normtensor(predict[0])
            img, pred, mak = tensor2np(input[0]), tensor2np(pred), tensor2np(mask[0])
    mean_iou = np.mean(iou)
    total_loss = running_loss/len(trainloader)
    return total_loss, mean_iou
