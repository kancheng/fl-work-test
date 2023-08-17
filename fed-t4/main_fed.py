#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

from libs import *
from med_fed_train import *
from utils.loss import *
from utils.dataset import *

# 引入 time 模組
import time
# 開始測量
s_start = time.time()
print('開始測時 : ')

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.log = True
    # setting
    loss_func_val = nn.CrossEntropyLoss()
    # load dataset and split users
    if args.methods == 'harmofl':
        args.all_clients = True
    
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        val_loaders = None
        test_loaders = []
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users, args.num_users_info)
            dict_users_test = mnist_iid(dataset_test, args.num_users, args.num_users_info)
            for idx in range(len(dict_users_test)):
                test_obj = DataLoader(DatasetSplit(dataset_test, dict_users_test[idx]), batch_size = 1)
                test_loaders.append(test_obj)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, args.num_users_info)
            dict_users_test = mnist_noniid(dataset_test, args.num_users, args.num_users_info)
            for idx in range(len(dict_users_test)):
                test_obj = DataLoader(DatasetSplit(dataset_test, dict_users_test[idx]), batch_size = 1)
                test_loaders.append(test_obj)
    elif args.dataset == 'emnist':
        trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
        dataset_train = datasets.EMNIST('./data/emnist/', split = 'digits', train=True, download=True, transform=trans_emnist)
        dataset_test = datasets.EMNIST('./data/emnist/', split = 'digits', train=False, download=True, transform=trans_emnist)
        val_loaders = None
        test_loaders = []
        if args.iid:
            dict_users = emnist_iid(dataset_train, args.num_users, args.num_users_info)
            dict_users_test = emnist_iid(dataset_test, args.num_users, args.num_users_info)
            for idx in range(len(dict_users_test)):
                test_obj = DataLoader(DatasetSplit(dataset_test, dict_users_test[idx]), batch_size = 1)
                test_loaders.append(test_obj)
        else:
             exit('Error: only consider IID setting in EMNIST')
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        val_loaders = None
        test_loaders = []
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users, args.num_users_info)
            dict_users_test = cifar_iid(dataset_test, args.num_users, args.num_users_info)
            for idx in range(len(dict_users_test)):
                test_obj = DataLoader(DatasetSplit(dataset_test, dict_users_test[idx]), batch_size = 1)
                test_loaders.append(test_obj)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users, args.num_users_info)
            dict_users_test = cifar_noniid(dataset_test, args.num_users, args.num_users_info)
            for idx in range(len(dict_users_test)):
                test_obj = DataLoader(DatasetSplit(dataset_test, dict_users_test[idx]), batch_size = 1)
                test_loaders.append(test_obj)
    elif args.dataset == 'cifar100':
        trans_cifar100 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar100)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar100)
        val_loaders = None
        test_loaders = []
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users, args.num_users_info)
            dict_users_test = cifar_iid(dataset_test, args.num_users, args.num_users_info)
            for idx in range(len(dict_users_test)):
                test_obj = DataLoader(DatasetSplit(dataset_test, dict_users_test[idx]), batch_size = 1)
                test_loaders.append(test_obj)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users, args.num_users_info)
            dict_users_test = cifar_noniid(dataset_test, args.num_users, args.num_users_info)
            for idx in range(len(dict_users_test)):
                test_obj = DataLoader(DatasetSplit(dataset_test, dict_users_test[idx]), batch_size = 1)
                test_loaders.append(test_obj)
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
        args.model = 'hfl'
        print('Camelyon17 Loading ...')
        # python main_fed.py --dataset camelyon17 --gpu 0 --local_bs 128
        # python main_fed.py --dataset camelyon17 --imbalance
        # python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 5 --gpu 0 --methods harmofl
        net_glob, loss_func_val, init_dataset, _1, _2, train_loaders, val_loaders, test_loaders = initialize_camelyon17(args)
        args.num_users = len(init_dataset)
        
        client_num = args.num_users
        client_weights = [1./client_num for i in range(client_num)]

        # exit('該功能正在測試中 ...')
    elif args.dataset == 'prostate':
        args.model = 'hfl'
        print('Prostate MRI Loading ...')
        # python main_fed.py --dataset prostate 
        net_glob, loss_func_val, init_dataset, _1, _2, train_loaders, val_loaders, test_loaders = initialize_prostate(args)
        args.num_users = len(init_dataset)
        client_num = args.num_users
        # exit('該功能正在測試中 ...')
    elif args.dataset == 'brainfets2022':
        args.model = 'hfl'
        print('FeTS2022 (brain) Loading ...')
        # python main_fed.py --dataset brainfets2022
        net_glob, loss_func_val, init_dataset, _1, _2, train_loaders, val_loaders, test_loaders = initialize_brain_fets(args)
        args.num_users = len(init_dataset)
        client_num = args.num_users
        # exit('該功能正在測試中 ...')
    else:
        exit('Error: unrecognized dataset')
    # Seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # PATH
    args.save_path = './save_model/checkpoint/{}/seed{}'.format(args.dataset, seed)
    exp_folder = 'HarmoFL_exp'
    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, 'HarmoFL')

    print('# Deive:', args.device)
    print('# Training Clients:{}'.format(args.dataset))

    log = args.log

    if log:
        log_path = args.save_path.replace('checkpoint', 'log')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'HarmoFL.log'), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        for k in list(vars(args).keys()):
            logfile.write('{}: {}\n'.format(k, vars(args)[k]))

    # federated client number
    client_num = args.num_users
    client_weights = [1./client_num for i in range(client_num)]

    # if args.dataset == 'salt' or args.dataset == 'medicalmnist':
    #     train_features, train_labels = next(iter(dataset_train))
    if args.model == 'mlp':
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
    elif args.dataset == 'camelyon17':
        print('Loading ...')
    elif args.dataset == 'prostate':
        print('Loading ...')
    elif args.dataset == 'brainfets2022':
        print('Loading ...')
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
        models = [net_glob for i in range(args.num_users)]
        print('INFO. : All clients - ', len(models))
    
    # Mes
    if args.methods == 'fedavg':
        print('INFO. : Methods - FedAvg')
    elif args.methods == 'harmofl':
        print('INFO. : Methods - HarmoFL')
    elif args.methods == 'feddc':
        print('INFO. : Methods - FedDC')

    if args.resume:
        checkpoint = torch.load(SAVE_PATH+'_latest', map_location=args.device)
        net_glob.load_state_dict(checkpoint['server_model'])
        for client_idx in range(client_num):
            models[client_idx].load_state_dict(checkpoint['server_model'])
            models[client_idx].to(args.device)

        if 'optim_0' in list(checkpoint.keys()):
            for client_idx in range(client_num):
                net_glob[client_idx].load_state_dict(checkpoint[f'optim_{client_idx}'])
        #for client_idx in range(client_num):
           # models[client_idx].to('cpu')

        best_epoch, best_acc  = checkpoint['best_epoch'], checkpoint['best_acc']
        start_iter = int(checkpoint['iter']) + 1

        print(f'Last time best:{best_epoch} acc :{best_acc}')
        print('Resume training from epoch {}'.format(start_iter))
    else:
        best_epoch = 0
        best_acc = [0. for j in range(client_num)]
        start_iter = 0

    for iter in range( start_iter, args.epochs):
        loss_locals = []
        if not args.all_clients:
            models = []
        # models = []

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:

            try:
                params = models[idx].parameters()
            except IndexError:
                params = net_glob.parameters()

            if args.dataset == 'prostate':
                optimizers = [WPOptim(params, base_optimizer=optim.Adam, lr=args.lr, alpha=args.alpha, weight_decay=1e-4) for idx in range(client_num)]
            elif args.dataset == 'brain':
                optimizers = [WPOptim(params, base_optimizer=optim.Adam, lr=args.lr, alpha=args.alpha, weight_decay=1e-4) for idx in range(client_num)]
            elif args.dataset == 'camelyon17':
                optimizers = [WPOptim(params, base_optimizer=optim.SGD, lr=args.lr, alpha=args.alpha, momentum=0.9, weight_decay=1e-4) for idx in range(client_num)]
            else :
                optimizers = [torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum) for idx in range(client_num)]
            if args.dataset == 'camelyon17' or args.dataset == 'prostate' or args.dataset == 'brainfets2022' :
                dataset_train = _1[idx]
                # local = LocalUpdate(args = args, dataset = dataset_train, 
                #                     idxs = None,
                #                     loss_func = loss_func_val, 
                #                     lu_loader = train_loaders[idx],
                #                     optimizer_op = optimizers)
                local = LocalUpdate(args = args, dataset = dataset_train, 
                                    idxs = None,
                                    loss_func = loss_func_val, 
                                    lu_loader = train_loaders[idx],
                                    optimizer = optimizers[idx])
            elif args.model == 'unet' and args.dataset == 'salt':
                loss_func_val = nn.BCEWithLogitsLoss()
                # local = LocalUpdate(args = args, dataset = dataset_train, 
                #                 idxs = dict_users[idx],
                #                 loss_func = loss_func_val,
                #                 optimizer_op = 'adam')
                local = LocalUpdate(args = args, dataset = dataset_train, 
                                idxs = dict_users[idx],
                                loss_func = loss_func_val,
                                optimizer = optimizers[idx])
            else:
                # local = LocalUpdate(args = args, dataset = dataset_train, 
                #                     idxs = dict_users[idx],
                #                     loss_func = loss_func_val,
                #                     optimizer_op = 'sgd')
                local = LocalUpdate(args = args, dataset = dataset_train, 
                                    idxs = dict_users[idx],
                                    loss_func = loss_func_val,
                                    optimizer = optimizers[idx])
            # 檢查 models 內有沒有初始化過
            if len(models)<idx:
                model, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                if args.all_clients:
                    models[idx] = copy.deepcopy(model)
                    # print('INFO. - models[idx] : ', models[idx])
                    # print('INFO. - type(models) : ', type(models))
                    # print('INFO. - len(models) : ', len(models))
                else:
                    models.append(copy.deepcopy(model))
                    # print('INFO. - models : ', models)
                    # print('INFO. - type(models) : ', type(models))
                    # print('INFO. - len(models) : ', len(models))
            else:
                model, loss = local.train(net=models[idx].to(args.device))
            # models.append(copy.deepcopy(model))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        with torch.no_grad():
            if args.methods == 'fedavg':
                w_glob = FedAvg(models)
                net_glob.load_state_dict(w_glob)
            elif args.methods == 'harmofl':
                net_glob, models = HarmoFL(net_glob, models, client_weights)
            # print loss
            loss_val_acc_listavg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter +1, loss_val_acc_listavg))
            loss_train.append(loss_val_acc_listavg)

            val_acc_list = [None for j in range(len(models))]
            print('============== Global Validation ==============')
            if args.log:
                    logfile.write('============== Global Validation ==============\n')
            for client_idx, model in enumerate(models):
                if val_loaders is None:
                    val_loss, val_acc = test_med(args, model, dataset_test, loss_func_val, args.device)
                else :
                    val_loss, val_acc = test_med(args, model, val_loaders[client_idx], loss_func_val, args.device)
                
                # MNIST ...
                #     acc_train, loss_train = test_img_classification(net_glob, dataset_train, args, type = 'ce')
                #     acc_test, loss_test = test_img_classification(net_glob, dataset_test, args, type = 'ce')
                #     print("Training accuracy: {:.2f}".format(acc_train))
                #     print("Testing accuracy: {:.2f}".format(acc_test))

                val_acc_list[client_idx] = val_acc
                # print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[client_idx], val_loss, val_acc))
                print(' Site :', client_idx)
                print(' Val  Loss:', val_loss)
                print(' Val  Acc:', val_acc)
                if args.log:
                    # logfile.write(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[client_idx], val_loss, val_acc))
                    # logfile.write(' Site: \n', datasets[client_idx])
                    logfile.write(' Val Loss:'+ str(val_loss))
                    logfile.write(' Val Acc:'+ str(val_acc))
                    
                    logfile.flush()
            # Test after each round
            print('============== Test ==============')
            if args.log:
                logfile.write('============== Test ==============\n')
            for client_idx, model in enumerate(models):
                _, test_acc = test_med(args, net_glob, test_loaders[client_idx], loss_func_val, args.device)
                print('Test site ', client_idx)
                print('Epoch:', str(iter))
                print('Test Acc:', test_acc)
                # if args.log:
                    # logfile.write(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, iter, test_acc))
            # Record best acc
            if np.mean(val_acc_list) > np.mean(best_acc):
                # print(client_idx)
                # for client_idx in range(client_num):
                for client_idx in range(len(val_acc_list)):
                    best_acc[client_idx] = val_acc_list[client_idx]
                    best_epoch = iter
                    best_changed=True
                print(' Best Epoch:' , best_epoch)
                # if args.log:
                #     logfile.write(' Best Epoch:'+ best_epoch)
            if best_changed:
                print(' Saving the local and server checkpoint to ', SAVE_PATH )
                if args.log: logfile.write(' Saving the local and server checkpoint to ' + SAVE_PATH)
              
                model_dicts = {'server_model': net_glob.state_dict(),
                                'best_epoch': best_epoch,
                                'best_acc': best_acc,
                                'iter': iter}
                
                for o_idx in range(len(val_acc_list)):
                # for o_idx in range(client_num):
                    model_dicts['optim_{}'.format(o_idx)] = optimizers[o_idx].state_dict()

                torch.save(model_dicts, SAVE_PATH)
                torch.save(model_dicts, SAVE_PATH+'_latest')
                best_changed = False
            else:
                # save the latest model
                print(' Saving the latest checkpoint to ', SAVE_PATH)
                if args.log: logfile.write(' Saving the latest checkpoint to '+ SAVE_PATH)
                
                model_dicts = {'server_model': net_glob.state_dict(),
                                'best_epoch': best_epoch,
                                'best_acc': best_acc,
                                'iter': iter}
                for o_idx in range(client_num):
                    model_dicts['optim_{}'.format(o_idx)] = optimizers[o_idx].state_dict()
                torch.save(model_dicts, SAVE_PATH+'_latest')
    # elif args.methods == 'feddc':
    # elif args.methods == 'feddyn':
    # elif args.methods == 'scaffold':
    # elif args.methods == 'fedprox':
    # elif args.methods == 'fedtp':
    # elif args.methods == 'fedsr':
    # elif args.methods == 'moon':
    # elif args.methods == 'fedbn':
    # elif args.methods == 'fedadam':
    # elif args.methods == 'fednova':
    # elif args.methods == 'groundtruth':

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid_{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, args.methods))

    # testing
    # net_glob.eval()
    # if args.model == 'unet' and args.dataset == 'salt':
    #     criterion = nn.BCEWithLogitsLoss()
    #     train_loss, train_iou = test_img_segmentation(net_glob, args.device, dataset_train, criterion)
    #     test_loss, test_iou = test_img_segmentation(net_glob, args.device, dataset_test, criterion)
    #     print(f'Train - Valid loss: {train_loss:.3f} | Train - Valid IoU: {train_iou:.3f} ')
    #     print(f'Test - Valid loss: {test_loss:.3f} | Test - Valid IoU: {test_iou:.3f} ')
    # else:
    #     acc_train, loss_train = test_img_classification(net_glob, dataset_train, args, type = 'ce')
    #     acc_test, loss_test = test_img_classification(net_glob, dataset_test, args, type = 'ce')
    #     print("Training accuracy: {:.2f}".format(acc_train))
    #     print("Testing accuracy: {:.2f}".format(acc_test))

# 結束測量
s_end = time.time()

# save Model
# torch.save(model.state_dict(), PATH)

# 輸出結果
print("執行時間 : %f 秒" % (s_end - s_start))

