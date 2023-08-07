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

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, emnist_iid, exter_iid
from utils.options import args_parser
from models.Update import LocalUpdate
# from models.Nets import MLP, CNNMnist, CNNCifar, CNNCifar100, Mnist_2NN, UNet, Emnist_NN, Salt_UNet
from models.Nets import *
from models.Fed import FedAvg
from models.test import test_img

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

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

from PIL import Image

from glob import glob
import sys
import random
from skimage.io import imread, imshow
from skimage.transform import resize

