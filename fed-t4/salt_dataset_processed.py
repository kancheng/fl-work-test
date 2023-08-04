#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10
import torch
import torch.utils.data as data
import os
import PIL.Image as Image
import cv2

class SaltDataset(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        # img_num = len(os.listdir(os.path.join(root, 'images')))
        img_list = os.listdir(os.path.join(root, 'images'))
        mask_list = os.listdir(os.path.join(root, 'masks'))

        imgs = []
        for file, mask in zip(img_list, mask_list):
            imgs.append([file, mask])

        self.imgs = imgs
        self.tempath_images = os.path.join(root, 'images')
        self.tempath_masks = os.path.join(root, 'masks')
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = cv2.imread(os.path.join(self.tempath_images, y_path))
        img_y = cv2.imread(os.path.join(self.tempath_masks, y_path))
        ## TEST !??
        if self.transform is not None:
            img_x = self.transform(img_x)
            print(img_x)
            img_x = torch.argmax(img_x, dim=1)
            print(img_x)
        if self.target_transform(img_y) is not None:
            print(img_y)
            img_y = self.target_transform(img_y)
            img_y = torch.argmax(img_y, dim=1)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)


class TestDataset(data.Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        # img_num = len(os.listdir(os.path.join(root, 'images')))
        img_list = os.listdir(os.path.join(root, 'images'))
        imgs = []
        for i, pic in enumerate(img_list):
            imgs.append(pic)
        self.imgs = imgs
        self.tempath_images = os.path.join(root, 'images')
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        print(x_path)
        img_x = cv2.imread(os.path.join(self.tempath_images, x_path))
        if self.transform is not None:
            img_x = self.transform(img_x)
        return img_x
    def __len__(self):
        return len(self.imgs)

