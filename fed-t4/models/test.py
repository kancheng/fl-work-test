#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.10

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img_classification(net_g, datatest, args, type = 'ce'):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    # data_loader = DataLoader(datatest, batch_size=args.bs)
    data_loader = DataLoader(datatest, batch_size=1)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
        # sum up batch loss
        if type == 'ce':
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        elif type == 'bce':
            # BCEWithLogitsLoss
            test_loss += F.binary_cross_entropy_with_logits(log_probs, target, reduction='sum').item()
        # test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)

    accuracy = 100.00 * correct / len(data_loader.dataset)
    # print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(data_loader.dataset), accuracy))
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

def tensor2np(tensor):
    tensor = tensor.squeeze().cpu()
    return tensor.detach().numpy()

def normtensor(tensor):
    tensor = torch.where(tensor<0., torch.zeros(1).cuda(), torch.ones(1).cuda())
    return tensor

def cal_iou(outputs, labels, SMOOTH=1e-6):
    with torch.no_grad():
        outputs = outputs.squeeze(1).bool()  # BATCH x 1 x H x W => BATCH x H x W
        labels = labels.squeeze(1).bool()
        intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
        union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
        iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
        # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return iou

    # return iou.cpu().detach().numpy()

def get_iou_score(outputs, labels):
    A = labels.squeeze(1).bool()
    pred = torch.where(outputs<0., torch.zeros(1).cuda(), torch.ones(1).cuda())
    B = pred.squeeze(1).bool()
    intersection = (A & B).float().sum((1,2))
    union = (A| B).float().sum((1, 2)) 
    iou = (intersection + 1e-6) / (union + 1e-6)  
    return iou.cpu().detach().numpy()

def test_img_segmentation(model, device, testloader, loss_function, best_iou):
    model.eval()
    running_loss = 0
    # mask_list, iou  = [], []
    iou = []
    with torch.no_grad():
        for i, (input, mask) in enumerate(testloader):
            input, mask = input.to(device), mask.to(device)

            predict = model(input)
            loss = loss_function(predict, mask)

            running_loss += loss.item()
            iou.append(get_iou_score(predict, mask).mean())

            # log the first image of the batch
            if ((i + 1) % 1) == 0:
                pred = normtensor(predict[0])
                img, pred, mak = tensor2np(input[0]), tensor2np(pred), tensor2np(mask[0])
                # mask_list.append(wandb_mask(img, pred, mak))

    test_loss = running_loss/len(testloader)
    mean_iou = np.mean(iou)
    # wandb.log({'Valid loss': test_loss, 'Valid IoU': mean_iou, 'Prediction': mask_list})
    
    # if mean_iou>best_iou:
    # # export to onnx + pt
    #     try:
    #         torch.onnx.export(model, input, SAVE_PATH + RUN_NAME + '.onnx')
    #         torch.save(model.state_dict(), SAVE_PATH + RUN_NAME + '.pth')
    #     except:
    #         print('Can export weights')

    return test_loss, mean_iou

