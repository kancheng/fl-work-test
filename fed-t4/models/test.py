#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args, type = 'ce'):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    print(len(data_loader))
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        print('D',len(data))
        print('T',len(target))
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
        # sum up batch loss
        print('XXXXXXXXXXXX',test_loss)
        if type == 'ce':
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        elif type == 'bce':
            # BCEWithLogitsLoss
            test_loss += F.binary_cross_entropy_with_logits(log_probs, target, reduction='sum').item()
        # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        print(correct)
    print('DLD',len(data_loader.dataset))
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    # print('DDDDDD',len(data_loader.dataset))
    print('TL', test_loss)
    print('AC', accuracy)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

