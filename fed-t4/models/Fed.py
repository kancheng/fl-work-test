#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


# communication - HarmoFL
# def HarmoFL(server_model, models, client_weights):
#     with torch.no_grad():
#         # aggregate params
#         for key in server_model.state_dict().keys():
#             temp = torch.zeros_like(server_model.state_dict()[key])
#             for client_idx in range(len(client_weights)):
#                 temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
#             server_model.state_dict()[key].data.copy_(temp)
#             for client_idx in range(len(client_weights)):
#                 models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
#             if 'running_amp' in key:
#                 # aggregate at first round only to save communication cost
#                 server_model.amp_norm.fix_amp = True
#                 for model in models:
#                     model.amp_norm.fix_amp = True
#     return server_model, models

# def FedXX(x):
#     x = x + 1
#     return x