#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNCifar100
from models.Fed import FedAvg
from models.test import test_img

#######

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

######

import os
import pandas as pd
import cv2
from tqdm import tqdm
from copy import deepcopy

import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler



