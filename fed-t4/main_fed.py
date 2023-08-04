#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

from libs import *
from salt_dataset_processed import *

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    if args.dataset == 'emnist':
        trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
        dataset_train = datasets.EMNIST('./data/emnist/', split = 'digits', train=True, download=True, transform=trans_emnist)
        dataset_test = datasets.EMNIST('./data/emnist/', split = 'digits', train=False, download=True, transform=trans_emnist)
        if args.iid:
            dict_users = emnist_iid(dataset_train, args.num_users)
        else:
             exit('Error: only consider IID setting in EMNIST')
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'cifar100':
        trans_cifar100 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar100)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar100)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR100')
    elif args.dataset == 'salt':
        # https://github.com/rabbitdeng/Unet-pytorch/blob/main/train.py
        batch_size = args.local_bs
        salt_train_dir = 'external/salt/train/'
        salt_test_dir = 'external/salt/test/images'
        x_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            # transforms.Resize([1024, 1024]),
            transforms.Resize([512, 512]),
            # transforms.Resize([256, 256]),
            # transforms.Grayscale(num_output_channels=1),
            # 标准化至[-1,1],规定均值和标准差
            transforms.Normalize([0.5], [0.5])  # torchvision.transforms.Normalize(mean, std, inplace=False)
        ])
        # mask 只需要转换为tensor
        y_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            # transforms.Resize([1024, 1024]),
            transforms.Resize([512, 512]),
            # transforms.Resize([256, 256]),
        ])
        
        train_dataset_pre = SaltDataset(salt_train_dir, transform=x_transform, target_transform=y_transform)
        # dataset_train = DataLoader(train_dataset_pre, batch_size = batch_size, shuffle=True)
        dataset_train = train_dataset_pre

        test_dataset_pre = SaltDataset(salt_train_dir, transform=x_transform, target_transform=y_transform)
        dataset_test = test_dataset_pre 
        if args.iid:
            dict_users = exter_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar100':
        net_glob = CNNCifar100(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == '2nn' and args.dataset == 'mnist':
        net_glob = Mnist_2NN(args=args).to(args.device)
    elif args.model == 'nn' and args.dataset == 'emnist':
        net_glob = Emnist_NN(args=args).to(args.device)
    elif args.model == 'unet' and args.dataset == 'salt':
        net_glob = Salt_UNet(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    # Mutli. Fed.    
    if args.methods == 'fedavg':
        for iter in range(args.epochs):
            loss_locals = []
            if not args.all_clients:
                w_locals = []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                # RuntimeError: Input type (torch.cuda.LongTensor) and weight type (torch.cuda.FloatTensor) should be the same
                if torch.cuda.is_available():
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device)).cuda()
                else:
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                # w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            # update global weights

            w_glob = FedAvg(w_locals)
            
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)
    # elif args.methods == 'harmofl':
    #     print('Testing ...')

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.methods))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

