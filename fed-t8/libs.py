#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
import copy
from copy import deepcopy
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms

from skimage.io import imread, imshow
from skimage.transform import resize

from utils.dataset import saltIDDataset, Camelyon17, Prostate, Brain
from utils.loss import DiceLoss, JointLoss
from models.general_models import DenseNet, UNet
# from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, emnist_iid, exter_iid
from utils.sampling import *
from utils.options import args_parser
from utils.weight_perturbation import *
from models.Update import LocalUpdate
# from models.Nets import MLP, CNNMnist, CNNCifar, CNNCifar100, Mnist_2NN, UNet, Emnist_NN, Salt_UNet
from models.Nets import *
from models.Fed import *
from models.test import *
from models.train import *
