#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.10

from libs import *

def trainImageFetch(images_id):
    image_train = np.zeros((images_id.shape[0], 101, 101), dtype=np.float32)
    mask_train = np.zeros((images_id.shape[0], 101, 101), dtype=np.float32)

    for idx, image_id in tqdm(enumerate(images_id), total=images_id.shape[0]):
        image_path = os.path.join(train_image_dir, image_id+'.png')
        mask_path = os.path.join(train_mask_dir, image_id+'.png')

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255

        image_train[idx] = image
        mask_train[idx] = mask
    
    return image_train, mask_train

def testImageFetch(test_id):
    image_test = np.zeros((len(test_id), 101, 101), dtype=np.float32)

    for idx, image_id in tqdm(enumerate(test_id), total=len(test_id)):
        image_path = os.path.join(test_image_dir, image_id+'.png')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        image_test[idx] = image

    return image_test

def do_resize2(image, mask, H, W):
    image = cv2.resize(image, dsize=(W,H))
    mask = cv2.resize(mask, dsize=(W,H))
    return image, mask

def do_center_pad(image, pad_left, pad_right):
    return np.pad(image, (pad_left, pad_right), 'edge')

def do_center_pad2(image, mask, pad_left, pad_right):
    image = do_center_pad(image, pad_left, pad_right)
    mask = do_center_pad(mask, pad_left, pad_right)
    return image, mask

class SaltDataset(Dataset):
    def __init__(self, image_list, mode, mask_list=None, fine_size=202, pad_left=0, pad_right=0):
        self.imagelist = image_list
        self.mode = mode
        self.masklist = mask_list
        self.fine_size = fine_size
        self.pad_left = pad_left
        self.pad_right = pad_right

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        image = deepcopy(self.imagelist[idx])

        if self.mode == 'train':
            mask = deepcopy(self.masklist[idx])
            label = np.where(mask.sum() == 0, 1.0, 0.0).astype(np.float32)

            if self.fine_size != image.shape[0]:
                image, mask = do_resize2(image, mask, self.fine_size, self.fine_size)

            if self.pad_left != 0:
                image, mask = do_center_pad2(image, mask, self.pad_left, self.pad_right)

            image = image.reshape(1, image.shape[0], image.shape[1])
            mask = mask.reshape(1, mask.shape[0], mask.shape[1])        

            return image, mask, label

        elif self.mode == 'val':
            mask = deepcopy(self.masklist[idx])

            if self.fine_size != image.shape[0]:
                image, mask = do_resize2(image, mask, self.fine_size, self.fine_size)

            if self.pad_left != 0:
                image = do_center_pad(image, self.pad_left, self.pad_right)

            image = image.reshape(1, image.shape[0], image.shape[1])
            mask = mask.reshape(1, mask.shape[0], mask.shape[1])    

            return image, mask

        elif self.mode == 'test':
            if self.fine_size != image.shape[0]:
                image = cv2.resize(image, dsize=(self.fine_size, self.fine_size))

            if self.pad_left != 0:
                image = do_center_pad(image, self.pad_left, self.pad_right)

            image = image.reshape(1, image.shape[0], image.shape[1])

            return image         