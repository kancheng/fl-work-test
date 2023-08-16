#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

# Reference
# https://github.com/TsingZ0/PFL-Non-IID/blob/master/dataset/utils/dataset_utils.py

import numpy as np
from torchvision import datasets, transforms

def crack(integer):
    start = int(np.sqrt(integer))
    factor = integer / start
    while not is_integer(factor):
        start += 1
        factor = integer / start
    return int(factor), start


def is_integer(number):
    if int(number) == number:
        return True
    else:
        return False
# crack(64000)  # 250, 256

def mnist_iid(dataset, num_users, num_users_info):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        if num_users_info:
            print('num - ', i, ' ; len ', len(dict_users[i]),' : ', dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users, num_users_info):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    # UserWarning: train_labels has been renamed targets
    labels = dataset.targets.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
            if num_users_info:
                print('num - ', i, ' ; len ', len(dict_users[i]),' : ', dict_users[i])
    return dict_users


def cifar_iid(dataset, num_users, num_users_info):
    """
    Sample I.I.D. client data from CIFAR10 &  CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        if num_users_info:
            print('num - ', i, ' ; len ', len(dict_users[i]),' : ', dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users, num_users_info):
    """
    Sample non-I.I.D client data from CIFAR10 &  CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.targets.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
            if num_users_info:
                print('num - ', i, ' ; len ', len(dict_users[i]),' : ', dict_users[i])
    return dict_users

def exter_iid(dataset, num_users, num_users_info):
    """
    Sample I.I.D. client data from External.
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        if num_users_info:
            print('num - ', i, ' ; len ', len(dict_users[i]),' : ', dict_users[i])
    return dict_users

def exter_noniid(dataset, num_users, num_users_info):
    """
    Sample non-I.I.D client data from CIFAR10 &  CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dataset_len_num = len(dataset)
    # key = np.arange()
    num_shards, num_imgs = crack(dataset_len_num)
    # num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs) # 根據大小建立索引
    images = dataset.images.numpy()
    idxs_labels = np.row_stack((idxs, images))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    # idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
            if num_users_info:
                print('num - ', i, ' ; len ', len(dict_users[i]),' : ', dict_users[i])
    return dict_users

def emnist_iid(dataset, num_users, num_users_info):
    """
    Sample I.I.D. client data from EMNIST dataset (Testing)
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        if num_users_info:
            print('num - ', i, ' ; len ', len(dict_users[i]),' : ', dict_users[i])
    return dict_users

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
