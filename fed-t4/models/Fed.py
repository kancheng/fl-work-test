#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

from libs import *

import copy
import torch
from torch import nn


def FedAvg(models):
    w_avg = copy.deepcopy(models[0].state_dict())
    for k in w_avg.keys():
        for i in range(1, len(models)):
            # state_dict() 模型內的權重參數
            w_avg[k] += models[i].state_dict()[k]
        w_avg[k] = torch.div(w_avg[k], len(models))
    return w_avg

def HarmoFL(server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        for key in server_model.state_dict().keys():
            # temp = torch.zeros_like(server_model.state_dict()[key])
            temp = torch.zeros_like(server_model.state_dict()[key]).long()
            
            for client_idx in range(len(models)):
                temp += (client_weights[client_idx] * models[client_idx].state_dict()[key].long())
                # temp += (client_weights[client_idx] * models[client_idx].state_dict()[key])

            server_model.state_dict()[key].data.copy_(temp)
            for client_idx in range(len(models)):
                models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
            if 'running_amp' in key:
                # aggregate at first round only to save communication cost
                server_model.amp_norm.fix_amp = True
                for model in models:
                    model.amp_norm.fix_amp = True
    return server_model, models

# methods 'feddc'

def FedDC(models):
    w_avg = 12
    return w_avg

# def FedXX(x):
#     x = x + 1
#     return x

# methods 'feddyn'
def FedDyn(models):
    w_avg = 12
    return w_avg

# methods 'scaffold'
def Scaffold(models):
    w_avg = 12
    return w_avg

# methods 'fedprox'
def FedProx(models):
    w_avg = 12
    return w_avg

# methods 'fedtp'
def FedTP(models):
    w_avg = 12
    return w_avg

# methods 'fedsr'
def FedSR(models):
    w_avg = 12
    return w_avg

# methods 'moon'
def Moon(models):
    w_avg = 12
    return w_avg

# methods 'fedbn'
def FedBN(models):
    w_avg = 12
    return w_avg

# methods 'fedadam'
def FedAdam(models):
    w_avg = 12
    return w_avg

# methods 'fednova'
def FedNova(models):
    w_avg = 12
    return w_avg

# methods 'groundtruth'
def GroundTruth(models):
    w_avg = 12
    return w_avg