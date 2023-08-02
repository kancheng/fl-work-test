################
####  TEST  ####
################
import externalconfig
# 外部資料測試
import torch
from torchvision import datasets, transforms
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader

# Required constants.
TRAIN_DIR = externalconfig.TRAIN_DIR
TEST_DIR = externalconfig.TEST_DIR
VALID_DIR = externalconfig.VALID_DIR
IMAGE_SIZE = externalconfig.IMAGE_SIZE
BATCH_SIZE = externalconfig.BATCH_SIZE
NUM_WORKERS = externalconfig.NUM_WORKERS

# 轉換 CIFAR-10 targets 格式的函數
def transform_cifar10_targets(target):
    # 在 CIFAR-10 中，目標是類別的索引（0 到 9）
    # 你可以根據需要進行其他處理，比如 one-hot 編碼等
    return target

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

def get_datasets():
    """
    Function to prepare the Datasets.
    Returns the training and validation datasets along 
    with the class names.
    """
    dataset_train = datasets.ImageFolder(
        TRAIN_DIR, 
        transform=(get_train_transform(IMAGE_SIZE))
    )
    dataset_valid = datasets.ImageFolder(
        VALID_DIR, 
        transform=(get_valid_transform(IMAGE_SIZE))
    )
    return dataset_train, dataset_valid, dataset_train.classes

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

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

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
        # print("111111111111111111111111111111111111111111111111111111111111")
        data = convert_image_folder_to_np(dataset_extra)
        target = np.array(targets)
        # print("222222222222222222222222222222222222222222222222222")

        if self.dataidxs is not None:
            # Test
            # # 看 self.dataidxs
            # print("--------------------------------------------------------")
            # print(type(self.dataidxs[1]))
            # print(type(self.dataidxs))
            # print(self.dataidxs)
            # print("--------------------------------------------------------")
            # print("--------------------------------------------------------")
            # print("--------------------------------------------------------")
            # #print(data)
            # print(type(data))
            # print("--------------------------------------------------------")
            # print("--------------------------------------------------------")
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
    




