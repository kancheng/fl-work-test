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

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users, args.num_users_info)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, args.num_users_info)
    elif args.dataset == 'emnist':
        trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
        dataset_train = datasets.EMNIST('./data/emnist/', split = 'digits', train=True, download=True, transform=trans_emnist)
        dataset_test = datasets.EMNIST('./data/emnist/', split = 'digits', train=False, download=True, transform=trans_emnist)
        if args.iid:
            dict_users = emnist_iid(dataset_train, args.num_users, args.num_users_info)
        else:
             exit('Error: only consider IID setting in EMNIST')
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users, args.num_users_info)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users, args.num_users_info)
            # exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'cifar100':
        trans_cifar100 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar100)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar100)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users, args.num_users_info)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users, args.num_users_info)
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
        # batch_size = 16
        batch_size = args.local_bs
        train_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_train, 
                                                batch_size=batch_size, 
                                                shuffle=True)
        dataset_train = salt_ID_dataset_train
        val_loader = torch.utils.data.DataLoader(dataset=salt_ID_dataset_val, 
                                                batch_size=batch_size, 
                                                shuffle=False)
        dataset_test = salt_ID_dataset_val
        if args.iid:
            dict_users = exter_iid(dataset_train, args.num_users, args.num_users_info)
        else:
            # dict_users = exter_noniid(dataset_train, args.num_users, args.num_users_info)
            exit('Error: only consider IID setting in the Salt Dataset.')
    elif args.dataset == 'medicalmnist':
        print('Medical MNIST Loading ...')
        # medical_mnist
        train_dir = './external/medical-mnist/medical_mnist_processed/train'
        valid_dir = './external/medical-mnist/medical_mnist_processed/test'
        image_size = 224 # Image size of resize when applying transforms.
        batch_size = args.local_bs # 64
        num_workers = 4 # Number of parallel processes for data preparation.
        dataset_train, dataset_valid, dataset_classes = get_datasets( train_dir, valid_dir, image_size)
        dataset_train, dataset_test = get_data_loaders(dataset_train, dataset_valid, batch_size, num_workers)
        # print(dataset_train)
        # print(dataset_test)
        if args.iid:
            dict_users = exter_iid(dataset_train, args.num_users, args.num_users_info)
        else:
            exit('Error: only consider IID setting in Medical MNIST.')
        # exit('該功能正在測試中 ...')
    elif args.dataset == 'camelyon17':
        print('Camelyon17 Loading ...')
        # python main_fed.py --dataset camelyon17
        # python main_fed.py --dataset camelyon17 --imbalance
        server_model, loss_fun, init_dataset, _1, _2, train_loaders, val_loaders, test_loaders = initialize_camelyon17(args)
        print(server_model)
        print(loss_fun)
        print(init_dataset)
        print(_1)
        print(_2)
        print(train_loaders)
        print(val_loaders)
        print(test_loaders)
        exit('該功能正在測試中 ...')
    elif args.dataset == 'prostate':
        print('Prostate MRI Loading ...')
        # python main_fed.py --dataset prostate 
        server_model, loss_fun, init_dataset, _1, _2, train_loaders, val_loaders, test_loaders = initialize_prostate(args)
        print(server_model)
        print(loss_fun)
        print(init_dataset)
        print(_1)
        print(_2)
        print(train_loaders)
        print(val_loaders)
        print(test_loaders)
        exit('該功能正在測試中 ...')
    elif args.dataset == 'brainfets2022':
        print('FeTS2022 (brain) Loading ...')
        # python main_fed.py --dataset brainfets2022
        server_model, loss_fun, init_dataset, _1, _2, train_loaders, val_loaders, test_loaders = initialize_brain_fets(args)
        print(server_model)
        print(loss_fun)
        print(init_dataset)
        print(_1)
        print(_2)
        print(train_loaders)
        print(val_loaders)
        print(test_loaders)
        exit('該功能正在測試中 ...')
    else:
        exit('Error: unrecognized dataset')
    
    if args.dataset == 'salt' or args.dataset == 'medicalmnist':
        train_features, train_labels = next(iter(dataset_train))
    else:
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
    elif args.model == 'medcnn' and args.dataset == 'medicalmnist':
        net_glob = MedicalMNISTCNN(args=args).to(args.device)
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
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
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
    elif args.methods == 'harmofl':
        for iter in range(args.epochs):
            loss_locals = []
            if not args.all_clients:
                w_locals = []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
            # update global weights
            w_glob = HarmoFL(w_locals)
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)
            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)
        exit('該功能正在測試中 ...')
    elif args.methods == 'feddc':
        print('Testing ...')
        exit('該功能正在測試中 ...')
    elif args.methods == 'feddyn':
        print('Testing ...')
        exit('該功能正在測試中 ...')
    elif args.methods == 'scaffold':
        print('Testing ...')
        exit('該功能正在測試中 ...')
    elif args.methods == 'fedprox':
        print('Testing ...')
        exit('該功能正在測試中 ...')
    elif args.methods == 'fedtp':
        print('Testing ...')
        exit('該功能正在測試中 ...')
    elif args.methods == 'fedsr':
        print('Testing ...')
        exit('該功能正在測試中 ...')
    elif args.methods == 'moon':
        print('Testing ...')
        exit('該功能正在測試中 ...')
    elif args.methods == 'fedbn':
        print('Testing ...')
        exit('該功能正在測試中 ...')
    elif args.methods == 'fedadam':
        print('Testing ...')
        exit('該功能正在測試中 ...')
    elif args.methods == 'fednova':
        print('Testing ...')
        exit('該功能正在測試中 ...')
    elif args.methods == 'groundtruth':
        print('Testing ...')
        exit('該功能正在測試中 ...')
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid_{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.methods))

    # testing
    net_glob.eval()
    if args.model == 'unet' and args.dataset == 'salt':
        criterion = nn.BCEWithLogitsLoss()
        train_loss, train_iou = test_img_segmentation(net_glob, args.device, dataset_train, criterion)
        test_loss, test_iou = test_img_segmentation(net_glob, args.device, dataset_test, criterion)
        print(f'Train - Valid loss: {train_loss:.3f} | Train - Valid IoU: {train_iou:.3f} ')
        print(f'Test - Valid loss: {test_loss:.3f} | Test - Valid IoU: {test_iou:.3f} ')
    else:
        acc_train, loss_train = test_img_classification(net_glob, dataset_train, args, type = 'ce')
        acc_test, loss_test = test_img_classification(net_glob, dataset_test, args, type = 'ce')
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))

