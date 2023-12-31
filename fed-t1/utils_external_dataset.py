import external_config

###################
### Testing !!! ###
###################

from utils_libs import *

EDIR = './Folder/External/'
# medical-mnist.
MED_MNIST_TRAIN_DIR = external_config.MED_MNIST_TRAIN_DIR
MED_MNIST_TEST_DIR = external_config.MED_MNIST_TEST_DIR
MED_MNIST_VALID_DIR = external_config.MED_MNIST_VALID_DIR
MED_MNIST_IMAGE_SIZE = external_config.MED_MNIST_IMAGE_SIZE
MED_MNIST_BATCH_SIZE = external_config.MED_MNIST_BATCH_SIZE
MED_MNIST_NUM_WORKERS = external_config.MED_MNIST_NUM_WORKERS

# salt
SALT_TRAIN_IMAGE_DIR = external_config.SALT_TRAIN_IMAGE_DIR
SALT_TRAIN_MASK_DIR = external_config.SALT_TRAIN_MASK_DIR
SALT_TEST_IMAGE_DIR = external_config.SALT_TEST_IMAGE_DIR

# setting val.

TRAIN_DIR = MED_MNIST_TRAIN_DIR
TEST_DIR = MED_MNIST_TEST_DIR
VALID_DIR = MED_MNIST_VALID_DIR
IMAGE_SIZE = MED_MNIST_IMAGE_SIZE
BATCH_SIZE = MED_MNIST_BATCH_SIZE
NUM_WORKERS = MED_MNIST_NUM_WORKERS



# Training transforms
def get_train_transform(IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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
def get_valid_transform(IMAGE_SIZE):
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
            )
    ])
    return valid_transform

def convert_image_folder_to_np( dataset, batch_size=32):
    # 创建一个DataLoader，用于逐批加载数据
    # RuntimeError: The size of tensor a (224) must match the size of tensor b (3) at non-singleton dimension 0
    # 解决方法，在DataLoader函数中加上一个参数drop_last=False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    # 创建一个空的列表来保存图像数据
    image_data_list = []
    # 逐批加载并转换图像，并保存到列表中
    for images, _ in dataloader:
        image_data_list.append(images.numpy())
    # 将列表中的图像数据堆叠为一个NumPy数组
    image_data = np.vstack(image_data_list)
    return image_data

def get_datasets_train():
    dataset_train = datasets.ImageFolder(
        TRAIN_DIR, 
        transform=(get_train_transform(IMAGE_SIZE))
    )
    return dataset_train

def get_datasets_test():
    dataset_valid = datasets.ImageFolder(
        TEST_DIR,
        transform=(get_valid_transform(IMAGE_SIZE))
    )

    return dataset_valid

def get_data_loaders(dataset_train, dataset_valid):
    """
    Prepares the training and validation data loaders.

    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.

    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader 

class EXTERNAL_truncated(data.DataLoader):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.target = self.__build_truncated_dataset__()
    def __build_truncated_dataset__(self):

        if self.train == True :
            dataset_extra = get_datasets_train()
        elif self.train == False :
            dataset_extra = get_datasets_test()
        # 處理 targets，得到 CIFAR-10 相同格式的 targets
        targets = torch.tensor(dataset_extra.targets)
        # data = dataset_extra
        data = convert_image_folder_to_np(dataset_extra)
        target = np.array(targets)
        if self.dataidxs is not None:
            # Test
            # # 看 self.dataidxs
            # print("--------------------------------------------------------")
            # print(type(self.dataidxs[1]))
            # print(type(self.dataidxs))
            # print(self.dataidxs)
            # print("--------------------------------------------------------")
            # # print(data)
            # print(type(data))
            # print("--------------------------------------------------------")
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0
    # 資料匯入格式
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # print("cifar10 img:", img)
        # print("cifar10 target:", target)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
    

# def load_external_data(EDIR):
def load_external_data(EDIR):
    transform = transforms.Compose([transforms.ToTensor()])
    external_train_ds = EXTERNAL_truncated(EDIR, train=True, transform=transform)
    external_test_ds = EXTERNAL_truncated(EDIR, train=False, transform=transform)
    # external_train_ds = EXTERNAL_truncated(train=True, transform=transform)
    # external_test_ds = EXTERNAL_truncated(train=False, transform=transform)
    X_train, y_train = external_train_ds.data, external_train_ds.target # X input data
    X_test, y_test = external_test_ds.data, external_test_ds.target # Y output data
    return (X_train, y_train, X_test, y_test)

class ExternalDatasetObject:
    def __init__(self, dataset, n_client, seed, rule, unbalanced_sgm=0, rule_arg='', data_path=''):
        self.dataset  = dataset
        self.n_client = n_client
        self.rule     = rule
        self.rule_arg = rule_arg
        self.seed     = seed
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        self.name = "%s_%d_%d_%s_%s" %(self.dataset, self.n_client, self.seed, self.rule, rule_arg_str)
        self.name += '_%f' %unbalanced_sgm if unbalanced_sgm!=0 else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = data_path
        self.set_data()
        
    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists('%sExternal/%s' %(self.data_path, self.name)):
            # Get Raw data
            # if self.dataset == 'brats20':
            #     print('test mes brats20.')

            # if self.dataset == 'camelyon17':
            #     print('test mes camelyon17.')

            # if self.dataset == 'fets2022':
            #     print('test mes fets2022.')

            if self.dataset == 'medical-mnist':
                print('test mes medical-mnist.')

                trn_load, tst_load = load_external_data(EDIR)
                # self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
                # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                # trnset = torchvision.datasets.MNIST(root='%sData/Raw' %self.data_path, 
                #                                     train=True , download=True, transform=transform)
                # tstset = torchvision.datasets.MNIST(root='%sData/Raw' %self.data_path, 
                #                                     train=False, download=True, transform=transform)
                # trn_load = torch.utils.data.DataLoader(trnset, batch_size=60000, shuffle=False, num_workers=1)
                # tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                # self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;

            # if self.dataset == 'oasis1':
            #     print('test mes oasis1.')

            # if self.dataset == 'prostate':
            #     print('test mes prostate.')
            if self.dataset == 'salt':
                print('test mes salt.')
                trn_load, tst_load = load_external_data(EDIR)
                # self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 1;
            
            # Shuffle Data
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(trn_y))
            trn_x = trn_x[rand_perm]
            trn_y = trn_y[rand_perm]
            
            self.trn_x = trn_x
            self.trn_y = trn_y
            self.tst_x = tst_x
            self.tst_y = tst_y
            
            
            ###
            n_data_per_clnt = int((len(trn_y)) / self.n_client)
            # Draw from lognormal distribution
            clnt_data_list = (np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=self.unbalanced_sgm, size=self.n_client))
            clnt_data_list = (clnt_data_list/np.sum(clnt_data_list)*len(trn_y)).astype(int)
            diff = np.sum(clnt_data_list) - len(trn_y)
            
            # Add/Subtract the excess number starting from first client
            if diff!= 0:
                for clnt_i in range(self.n_client):
                    if clnt_data_list[clnt_i] > diff:
                        clnt_data_list[clnt_i] -= diff
                        break
            ###     
            
            if self.rule == 'Drichlet':
                cls_priors   = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls,size=self.n_client)
                prior_cumsum = np.cumsum(cls_priors, axis=1)
                idx_list = [np.where(trn_y==i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]

                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
    
                while(np.sum(clnt_data_list)!=0):
                    curr_clnt = np.random.randint(self.n_client)
                    # If current node is full resample a client
                    print('Remaining Data: %d' %np.sum(clnt_data_list))
                    if clnt_data_list[curr_clnt] <= 0:
                        continue
                    clnt_data_list[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if trn_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            continue
                        cls_amount[cls_label] -= 1
                        
                        clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
                        clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]

                        break
                
                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)
                
                cls_means = np.zeros((self.n_client, self.n_cls))
                for clnt in range(self.n_client):
                    for cls in range(self.n_cls):
                        cls_means[clnt,cls] = np.mean(clnt_y[clnt]==cls)
                prior_real_diff = np.abs(cls_means-cls_priors)
                print('--- Max deviation from prior: %.4f' %np.max(prior_real_diff))
                print('--- Min deviation from prior: %.4f' %np.min(prior_real_diff))
            
            elif self.rule == 'iid' and self.dataset == 'CIFAR100' and self.unbalanced_sgm==0:
                assert len(trn_y)//100 % self.n_client == 0 
                
                # create perfect IID partitions for cifar100 instead of shuffling
                idx = np.argsort(trn_y[:, 0])
                n_data_per_clnt = len(trn_y) // self.n_client
                # clnt_x dtype needs to be float32, the same as weights
                clnt_x = np.zeros((self.n_client, n_data_per_clnt, 3, 32, 32), dtype=np.float32)
                clnt_y = np.zeros((self.n_client, n_data_per_clnt, 1), dtype=np.float32)
                trn_x = trn_x[idx] # 50000*3*32*32
                trn_y = trn_y[idx]
                n_cls_sample_per_device = n_data_per_clnt // 100
                for i in range(self.n_client): # devices
                    for j in range(100): # class
                        clnt_x[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :, :, :] = trn_x[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :, :, :]
                        clnt_y[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :] = trn_y[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :] 
            
            
            elif self.rule == 'iid':
                
                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ]
            
                clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(clnt_data_list)))
                for clnt_idx_ in range(self.n_client):
                    clnt_x[clnt_idx_] = trn_x[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                    clnt_y[clnt_idx_] = trn_y[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                
                
                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)

            
            self.clnt_x = clnt_x; self.clnt_y = clnt_y

            self.tst_x  = tst_x;  self.tst_y  = tst_y
            
            # Save data
            os.mkdir('%sData/%s' %(self.data_path, self.name))
            
            np.save('%sData/%s/clnt_x.npy' %(self.data_path, self.name), clnt_x)
            np.save('%sData/%s/clnt_y.npy' %(self.data_path, self.name), clnt_y)

            np.save('%sData/%s/tst_x.npy'  %(self.data_path, self.name),  tst_x)
            np.save('%sData/%s/tst_y.npy'  %(self.data_path, self.name),  tst_y)

        else:
            print("Data is already downloaded")
            self.clnt_x = np.load('%sData/%s/clnt_x.npy' %(self.data_path, self.name),allow_pickle=True)
            self.clnt_y = np.load('%sData/%s/clnt_y.npy' %(self.data_path, self.name),allow_pickle=True)
            self.n_client = len(self.clnt_x)

            self.tst_x  = np.load('%sData/%s/tst_x.npy'  %(self.data_path, self.name),allow_pickle=True)
            self.tst_y  = np.load('%sData/%s/tst_y.npy'  %(self.data_path, self.name),allow_pickle=True)
            
            if self.dataset == 'mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'CIFAR10':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            if self.dataset == 'fashion_mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'emnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
                
             
        print('Class frequencies:')
        count = 0
        for clnt in range(self.n_client):
            print("Client %3d: " %clnt + 
                  ', '.join(["%.3f" %np.mean(self.clnt_y[clnt]==cls) for cls in range(self.n_cls)]) + 
                  ', Amount:%d' %self.clnt_y[clnt].shape[0])
            count += self.clnt_y[clnt].shape[0]
        
        
        print('Total Amount:%d' %count)
        print('--------')

        print("      Test: " + 
              ', '.join(["%.3f" %np.mean(self.tst_y==cls) for cls in range(self.n_cls)]) + 
              ', Amount:%d' %self.tst_y.shape[0])
