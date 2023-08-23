#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print( 'INFO. : THE BASE PATH : ', base_path)
sys.path.append(base_path)

from libs import *

def train_perturbation(args, model, data_loader, optimizer, loss_fun, device):
    model.to(device)
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    train_acc = 0.
    segmentation = model.__class__.__name__ == 'UNet'

    for step, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()

        #data = data.to(device)
        #target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()

        if segmentation:
            train_acc += DiceLoss().dice_coef(output, target).item()
        else:
            total += target.size(0)
            pred = output.data.max(1)[1]
            batch_correct = pred.eq(target.view(-1)).sum().item()
            correct += batch_correct
            if step % math.ceil(len(data_loader)*0.2) == 0:
                print(' [Step-{}|{}]| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(step, len(data_loader), loss.item(), batch_correct/target.size(0)), end='\r')

        loss.backward()
        optimizer.generate_delta(zero_grad=True)
        loss_fun(model(data), target).backward()
        optimizer.step(zero_grad=True)

    loss = loss_all / len(data_loader)
    acc = train_acc/ len(data_loader) if segmentation else correct/total

    # model.to('cpu')
    return loss, acc

def test_med(args, model, data_loader, loss_fun, device):
    if isinstance(data_loader,torch.utils.data.DataLoader) != True:
        data_loader = torch.utils.data.DataLoader(data_loader)
    model.to(device)
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    test_acc = 0.
    segmentation = model.__class__.__name__ == 'UNet'

    with torch.no_grad():
        for step, (data, target) in enumerate(data_loader):
            data = data.to(args.device)
            
            target = target.to(args.device)
            # output 模型的結果
            output = model(data)
            # print('output',output)
            loss = loss_fun(output, target)
            loss_all += loss.item()
            # print('loss_all',loss_all)
            if segmentation:
                test_acc += DiceLoss().dice_coef(output, target).item()
            else:
                total += target.size(0)
                pred = output.data.max(1)[1]
                pred = output.data.max(0)[0]
                # torch.max()这个函数返回的是两个值，第一个值是具体的value（我们用下划线_表示），第二个值是value所在的index（也就是predicted）。
                # https://pytorch.org/docs/stable/generated/torch.max.html
                # print('CCCCCCCCCCCCCCCCCCCC', pred)
                batch_correct = pred.eq(target.view(-1)).sum().item()
                # 而 torch.eq() 函数就是用来比较对应位置数字，相同则为1，否则为0，输出与那两个tensor大小相同，并且其中只有1和0。
                # https://pytorch.org/docs/stable/generated/torch.eq.html
                # torch.eq().sum()就是将所有值相加，但得到的仍是tensor.
                # torch.eq().sum()得到的结果就是[2]。
                # torch.eq().sum().item()得到值2。
                # Pytorch 中的 view 函数作用是将张量铺平，
                # print('batch_correct',batch_correct)
                correct += batch_correct
                # print('correct += batch_correct',batch_correct)
                # print('target.size(0)',target.size(0))
                if step % math.ceil(len(data_loader)*0.2) == 0:
                    print(' [Step-{}|{}]| Test Acc: {:.4f}'.format(step, len(data_loader), batch_correct/target.size(0)), end='\r')

    loss = loss_all / len(data_loader)
    # print('loss = loss_all / len(data_loader)',loss)
    acc = test_acc/ len(data_loader) if segmentation else correct/total
    return loss, acc

def initialize_camelyon17(args):
    train_loaders, test_loaders = [], []
    val_loaders = []
    trainsets, testsets = [], []
    valsets = []
    # camelyon17
    args.lr = 1e-3
    model = DenseNet(input_shape=[3,96,96]) # Dense121
    loss_fun = nn.CrossEntropyLoss()
    # sites = ['1', '2', '3', '4', '5']
    # sites = ['1', '2', '3']
    sites = ['1']
    for site in sites:
        trainset = Camelyon17(site=site, split='train', transform=transforms.ToTensor())
        testset = Camelyon17(site=site, split='test', transform=transforms.ToTensor())
        val_len = math.floor(len(trainset)*0.2)
        train_idx = list(range(len(trainset)))[:-val_len]
        val_idx = list(range(len(trainset)))[-val_len:]
        valset   = torch.utils.data.Subset(trainset, val_idx)
        trainset = torch.utils.data.Subset(trainset, train_idx)
        print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
        trainsets.append(trainset)
        valsets.append(valset)
        testsets.append(testset)
    min_data_len = min([len(s) for s in trainsets])
    for idx in range(len(trainsets)):
        if args.imbalance:
            # print('============== imbalance true ==============')
            trainset = trainsets[idx]
            valset = valsets[idx]
            testset = testsets[idx]
            # print('imbalance true - trainset', len(trainset))
            # print('imbalance true - valset',len(valset))
            # print('imbalance true - testset',len(testset))
        else:
            # print('============== imbalance false ==============')
            # imbalance false 會根據擁有最小 trainset 的 Client 將每個設定到最小
            trainset = torch.utils.data.Subset(trainsets[idx], list(range(int(min_data_len))))
            valset = valsets[idx]
            testset = testsets[idx]
            # print('imbalance false - trainset',len(trainset))
            # print('imbalance false - valset',len(valset))
            # print('imbalance false - testset',len(testset))

        train_loaders.append(torch.utils.data.DataLoader(trainset, batch_size=args.local_bs, shuffle=True))
        val_loaders.append(torch.utils.data.DataLoader(valset, batch_size=args.local_bs, shuffle=False))
        test_loaders.append(torch.utils.data.DataLoader(testset, batch_size=args.local_bs, shuffle=False))
    return model, loss_fun, sites, trainsets, testsets, train_loaders, val_loaders, test_loaders

def initialize_prostate(args):
    train_loaders, test_loaders = [], []
    val_loaders = []
    trainsets, testsets = [], []
    valsets = []
    # prostate
    args.lr = 1e-4
    args.iters = 500
    model = UNet(input_shape=[3, 384, 384])
    loss_fun = JointLoss()
    sites = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    for site in sites:
        trainset = Prostate(site=site, split='train', transform=transform)
        valset = Prostate(site=site, split='val', transform=transform)
        testset = Prostate(site=site, split='test', transform=transform)

        print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
        trainsets.append(trainset)
        valsets.append(valset)
        testsets.append(testset)

    min_data_len = min([len(s) for s in trainsets])
    for idx in range(len(trainsets)):
        if args.imbalance:
            trainset = trainsets[idx]
            valset = valsets[idx]
            testset = testsets[idx]
        else:
            trainset = torch.utils.data.Subset(trainsets[idx], list(range(int(min_data_len))))
            valset = valsets[idx]
            testset = testsets[idx]

        train_loaders.append(torch.utils.data.DataLoader(trainset, batch_size=args.local_bs, shuffle=True))
        val_loaders.append(torch.utils.data.DataLoader(valset, batch_size=args.local_bs, shuffle=False))
        test_loaders.append(torch.utils.data.DataLoader(testset, batch_size=args.local_bs, shuffle=False))
    return model, loss_fun, sites, trainsets, testsets, train_loaders, val_loaders, test_loaders

def initialize_brain_fets(args):
    train_loaders, test_loaders = [], []
    val_loaders = []
    trainsets, testsets = [], []
    valsets = []
    # brain
    args.lr = 1e-4
    args.iters = 500
    model = UNet(input_shape=[3, 240, 240])
    loss_fun = JointLoss()
    sites = ['1', '4', '5', '6', '13', '16', '18', '20', '21']
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    for site in sites:
        trainset = Brain(site=site, split='train', transform=transform)
        valset = Brain(site=site, split='val', transform=transform)
        testset = Brain(site=site, split='test', transform=transform)

        print(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
        trainsets.append(trainset)
        valsets.append(valset)
        testsets.append(testset)

    min_data_len = min([len(s) for s in trainsets])
    for idx in range(len(trainsets)):
        if args.imbalance:
            trainset = trainsets[idx]
            valset = valsets[idx]
            testset = testsets[idx]
        else:
            trainset = torch.utils.data.Subset(trainsets[idx], list(range(int(min_data_len))))
            valset = valsets[idx]
            testset = testsets[idx]

        train_loaders.append(torch.utils.data.DataLoader(trainset, batch_size=args.local_bs, shuffle=True))
        val_loaders.append(torch.utils.data.DataLoader(valset, batch_size=args.local_bs, shuffle=False))
        test_loaders.append(torch.utils.data.DataLoader(testset, batch_size=args.local_bs, shuffle=False))
    return model, loss_fun, sites, trainsets, testsets, train_loaders, val_loaders, test_loaders