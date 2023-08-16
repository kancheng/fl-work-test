#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

from libs import *
from med_fed_train import *
from utils.loss import *
from utils.dataset import *

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    torch.manual_seed(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        img_size = dataset_train[0][0].shape
    elif args.dataset == 'emnist':
        trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
        dataset_train = datasets.EMNIST('./data/emnist/', split = 'digits', train=True, download=True, transform=trans_emnist)
        img_size = dataset_train[0][0].shape
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, transform=transform, target_transform=None, download=True)
        img_size = dataset_train[0][0].shape
    elif args.dataset == 'cifar100':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, transform=transform, target_transform=None, download=True)
        img_size = dataset_train[0][0].shape
    elif args.dataset == 'salt':
        path_train = './external/salt/train'
        path_test = './external/salt/test'
        # Set some parameters# Set s 
        im_width = 128
        im_height = 128
        im_chan = 1
        train_path_images = os.path.abspath(path_train + "/images/")
        train_path_masks = os.path.abspath(path_train + "/masks/")

        test_path_images = os.path.abspath(path_test + "/images/")
        test_path_masks = os.path.abspath(path_test + "/masks/")
        train_ids = next(os.walk(train_path_images))[2]
        test_ids = next(os.walk(test_path_images))[2]
        # Get and resize train images and masks
        X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
        Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool_)
        print('Getting and resizing train images and masks ... ')
        sys.stdout.flush()
        for n, id_ in enumerate(train_ids):
            img = cv2.imread(path_train + '/images/' + id_, cv2.IMREAD_UNCHANGED)
            x = resize(img, (128, 128, 1), mode='constant', preserve_range=True)
            X_train[n] = x
            mask = cv2.imread(path_train + '/masks/' + id_, cv2.IMREAD_UNCHANGED)
            Y_train[n] = resize(mask, (128, 128, 1), 
                                mode='constant', 
                                preserve_range=True)
        print('Salt Done!')
        X_train_shaped = X_train.reshape(-1, 1, 128, 128)/255
        Y_train_shaped = Y_train.reshape(-1, 1, 128, 128)
        X_train_shaped = X_train_shaped.astype(np.float32)
        Y_train_shaped = Y_train_shaped.astype(np.float32)
        torch.cuda.manual_seed_all(4200)
        np.random.seed(133700)
        indices = list(range(len(X_train_shaped)))
        np.random.shuffle(indices)

        val_size = 1/10
        split = np.int_(np.floor(val_size * len(X_train_shaped)))

        train_idxs = indices[split:]
        val_idxs = indices[:split]
        salt_ID_dataset_train = saltIDDataset(X_train_shaped[train_idxs], 
                                      train=True, 
                                      preprocessed_masks=Y_train_shaped[train_idxs])
        salt_ID_dataset_val = saltIDDataset(X_train_shaped[val_idxs], 
                                            train=True, 
                                            preprocessed_masks=Y_train_shaped[val_idxs])
        batch_size = args.local_bs
        train_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_train, 
                                                batch_size=batch_size, 
                                                shuffle=True)
        # done
        dataset_train_pro = train_loader
        dataset_train = salt_ID_dataset_train
        val_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_val, 
                                                batch_size=batch_size, 
                                                shuffle=False)
    elif args.dataset == 'medicalmnist':
        train_dir = './external/medical-mnist/medical_mnist_processed/train'
        valid_dir = './external/medical-mnist/medical_mnist_processed/test'
        image_size = 224 # Image size of resize when applying transforms.
        batch_size = args.local_bs # 64
        num_workers = 4 # Number of parallel processes for data preparation.
        dataset_train, dataset_valid, dataset_classes = get_datasets( train_dir, valid_dir, image_size)
        dataset_train, dataset_test = get_data_loaders(dataset_train, dataset_valid, batch_size, num_workers)
    else:
        exit('Error: unrecognized dataset')

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
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # training
    if args.model == 'unet' and args.dataset == 'salt':
        optimizer = torch.optim.Adam(net_glob.parameters(), lr=args.lr)
    else :
        optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)

    # train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)
    train_loader = DataLoader(dataset_train, batch_size=args.local_bs, shuffle=True)
    


    list_loss = []
    net_glob.train()
    for epoch in range(args.epochs):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            if args.model == 'unet' and args.dataset == 'salt':
                # loss = nn.BCEWithLogitsLoss()(output, target)
                loss = F.binary_cross_entropy_with_logits(output, target)
            else:
                loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)

    # plot loss
    plt.figure()
    plt.plot(range(len(list_loss)), list_loss)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.savefig('./log/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))

    # testing
    if args.dataset == 'mnist':
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, transform=transform, target_transform=None, download=True)
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    elif args.dataset == 'cifar100':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, transform=transform, target_transform=None, download=True)
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    elif args.dataset == 'emnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
        dataset_test = datasets.EMNIST('./data/emnist/', split = 'digits', train=False, download=True, target_transform=None,  transform=transform)
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    elif args.dataset == 'salt':
        path_test = './external/salt/test'
        dataset_test = salt_ID_dataset_val
    else:
        exit('Error: unrecognized dataset')

    print('test on', len(dataset_test), 'samples')
    if args.dataset == 'salt':
        # test_acc, test_loss = test_local(net_glob, val_loader , type='bce')
        criterion = nn.BCEWithLogitsLoss()
        test_loss, test_iou = test_local_segmentation(net_glob, args.device, dataset_test, criterion)
        print(f'Valid loss: {test_loss:.3f} | Valid IoU: {test_iou:.3f} ')
    else :
        test_acc, test_loss = test_local_classification(net_glob, test_loader, args, type='ce')

