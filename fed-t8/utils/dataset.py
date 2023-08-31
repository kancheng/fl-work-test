#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image
import SimpleITK as sitk
import random
import cv2
import torch
from torchvision import datasets, transforms
import torchvision.transforms as transforms

# np.long -> np.longlong

# Training transforms
def get_train_transform(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
            )
    ])
    return train_transform

# Validation transforms
def get_valid_transform(image_size):
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
            )
    ])
    return valid_transform

def get_datasets( train_dir, valid_dir, image_size):
    """
    Function to prepare the Datasets.

    Returns the training and validation datasets along 
    with the class names.
    """
    dataset_train = datasets.ImageFolder(
        train_dir, 
        transform=(get_train_transform(image_size))
    )
    dataset_valid = datasets.ImageFolder(
        valid_dir, 
        transform=(get_valid_transform(image_size))
    )
    return dataset_train, dataset_valid, dataset_train.classes

def get_data_loaders(dataset_train, dataset_valid, bs, nw):
    """
    Prepares the training and validation data loaders.

    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.

    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=bs, 
        shuffle=True, num_workers=nw
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=bs, 
        shuffle=False, num_workers=nw
    )
    return train_loader, valid_loader 

def convert_from_nii_to_png(img):
    high = np.quantile(img,0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  
    newimg = (newimg * 255).astype(np.uint8)
    return newimg

class saltIDDataset(torch.utils.data.Dataset):
    def __init__(self,preprocessed_images, train=True, preprocessed_masks=None):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.train = train
        self.images = preprocessed_images
        if self.train:
            self.masks = preprocessed_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = None
        if self.train:
           mask = self.masks[idx]
        return (image ,mask)

class Camelyon17(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        assert split in ['train', 'test']
        assert int(site) in [1,2,3,4,5] # five hospital

        base_path = base_path if base_path is not None else './external/camelyon17'
        self.base_path = base_path

        data_dict = np.load('./external/camelyon17/data.pkl', allow_pickle=True)
        self.paths, self.labels = data_dict[f'hospital{site}'][f'{split}']

        self.transform = transform
        self.labels = self.labels.astype(np.longlong).squeeze()

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class Prostate(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        channels = {'BIDMC':3, 'HK':3, 'I2CVB':3, 'ISBI':3, 'ISBI_1.5':3, 'UCL':3}
        assert site in list(channels.keys())
        self.split = split

        base_path = base_path if base_path is not None else'./external/prostate/'

        sitedir = os.path.join(base_path, site)
        imgsdir = os.path.join(sitedir, 'images')
        labelsdir = os.path.join(sitedir, 'labels')

        ossitedir = os.listdir(imgsdir) # np.load("./external/prostate/{}-dir.npy".format(site)).tolist()
        # print(ossitedir)
        np.random.seed(2023)  # 先定义一个随机数种子
        lens = len(ossitedir)
        idx = np.random.rand(lens)

        idx_train = []
        idx_val = []
        idx_test = []
        for i in range (lens):
            if idx[i] < 0.65:
                idx_train.append(i)
            elif idx[i] <= 0.8:
                idx_val.append(i)
            else:
                idx_test.append(i)
        
        
        if(split=='train'):
            index = idx_train
        elif(split=='val'):
            index = idx_val
        else:
            index = idx_test
        
        if len(index) == 0:
            print('dataset split error')
        
        nums = 0
        images, labels = [], []
        for i in index:
            sample = ossitedir[i]
            # print(sample)

            # if os.path.getsize(sampledir) < 1024 * 1024 and sampledir.endswith("_segmentation.nii.gz"):
            imgdir = os.path.join(imgsdir, sample)
            labeldir = os.path.join(labelsdir, sample)
            label_v = sitk.ReadImage(labeldir)
            image_v = sitk.ReadImage(imgdir)
            label_v = sitk.GetArrayFromImage(label_v)
            label_v[label_v > 1] = 1
            image_v = sitk.GetArrayFromImage(image_v)
            image_v = convert_from_nii_to_png(image_v)

            for i in range(1, label_v.shape[0] - 1):
                    label = np.array(label_v[i, :, :])
                    if (np.all(label == 0)):
                        continue
                    image = np.array(image_v[i-1:i+2, :, :])
                    image = np.transpose(image,(1,2,0))
                    
                    labels.append(label)
                    images.append(image)
        
        labels = np.array(labels).astype(int)
        images = np.array(images)
        
        print(site, split, images.shape)

        self.images, self.labels = images, labels
        self.transform = transform
        self.channels = channels[site]
        self.labels = self.labels.astype(np.longlong).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            if self.split == 'train':
                R1 = RandomRotate90()
                image, label = R1(image, label)
                R2 = RandomFlip()
                image, label = R2(image, label)

            image = np.transpose(image,(2, 0, 1))
            image = torch.Tensor(image)
            
            label = self.transform(label)

        return image, label

class Brain(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        channels = {'1':3, '4':3, '5':3, '6':3, '13':3, '16':3, '18':3, '20':3, '21':3}
        # channels = {'1':3, '4':3, '5':3, '6':3}
        assert site in list(channels.keys())
        self.split = split

        base_path = base_path if base_path is not None else'./external/fets2022/'

        sitedir = os.path.join(base_path, site)
        imgsdir = os.path.join(sitedir, 'images')
        labelsdir = os.path.join(sitedir, 'labels')

        ossitedir = os.listdir(imgsdir) # np.load("../external/fets2022/{}-dir.npy".format(site)).tolist()
        # print(ossitedir)
        np.random.seed(2023)  # 先定义一个随机数种子
        lens = len(ossitedir)
        idx = np.random.rand(lens)

        idx_train = []
        idx_val = []
        idx_test = []
        for i in range (lens):
            if idx[i] < 0.65:
                idx_train.append(i)
            elif idx[i] <= 0.8:
                idx_val.append(i)
            else:
                idx_test.append(i)
        
        if(split=='train'):
            index = idx_train
        elif(split=='val'):
            index = idx_val
        else:
            index = idx_test
        
        if len(index) == 0:
            print('dataset split error')
        
        nums = 0
        images, labels = [], []
        for i in index:
            sample = ossitedir[i]
            # print(sample)

            # if os.path.getsize(sampledir) < 1024 * 1024 and sampledir.endswith("_segmentation.nii.gz"):
            imgdir = os.path.join(imgsdir, sample)
            labeldir = os.path.join(labelsdir, sample)
            label_v = sitk.ReadImage(labeldir)
            image_v = sitk.ReadImage(imgdir)
            label_v = sitk.GetArrayFromImage(label_v)
            # label_v[label_v != 4] = 0
            # label_v[label_v == 4] = 1
            label_v[label_v == 4] = 1
            label_v[label_v != 1] = 0
            image_v = sitk.GetArrayFromImage(image_v)
            image_v = convert_from_nii_to_png(image_v)

            for i in range(1, label_v.shape[0] - 1):
                label = np.array(label_v[i, :, :])
                if (np.all(label == 0)) and i%5 != 0:
                    continue
                image = np.array(image_v[i-1:i+2, :, :])
                image = np.transpose(image,(1,2,0))
                
                labels.append(label)
                images.append(image)
        
        labels = np.array(labels).astype(int)
        images = np.array(images)
        
        print(site, split, images.shape)

        self.images, self.labels = images, labels
        self.transform = transform
        self.channels = channels[site]
        self.labels = self.labels.astype(np.longlong).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            if self.split == 'train':
                R1 = RandomRotate90()
                image, label = R1(image, label)
                R2 = RandomFlip()
                image, label = R2(image, label)

            image = np.transpose(image,(2, 0, 1))
            image = torch.Tensor(image)
            
            label = self.transform(label)

        return image, label

class RandomRotate90:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()

class RandomFlip:
    def __init__(self, prob=0.75):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)

        return  img, mask

if __name__=='__main__':
    exit()




