###################################################################################################
#
# Copyright (C) 2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Classes and functions used to create RFSegment dataset.
"""
import copy
import csv
import os
import sys

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

import ai8x


class RFSegment(Dataset):
    """
    Possible class selections:
    0: None
    1: LTE
    2: NR
    """

    class_dict = {'None': 0, 'LTE': 1, 'NR': 2}

    def __init__(self, root_dir, d_type, classes=None, transform=None,
                 im_size=(256, 256)):
        self.transform = transform
        self.classes = classes

        img_dims = list(im_size)
        img_folder = os.path.join(root_dir, d_type)
        lbl_folder = os.path.join(root_dir, d_type + '_labels')
        self.class_dict_file = os.path.join(root_dir, 'class_dict.csv')
        self.img_list = []
        self.lbl_list = []
        # List of filenames for debugging
        self.file_list = []

        self.label_mask_dict = {}
        self.__create_mask_dict(img_dims)

        img_file_list = sorted(os.listdir(img_folder))

        for img_file in img_file_list:
            img = Image.open(os.path.join(img_folder, img_file))
            data_name = os.path.splitext(img_file)[0]
            lbl = Image.open(os.path.join(lbl_folder, 'mask_'+data_name+'.hdf.png'))
            if im_size == [352, 352]:
                (img, lbl) = self.pad_image_and_label(img, lbl)
            img = np.asarray(img)
            img = RFSegment.normalize(img.astype(np.float32))
            lbl_gray = np.asarray(lbl)
            lbl_rgb = np.empty((lbl_gray.shape[0],lbl_gray.shape[1],3))
            for i in range(3):
                lbl_rgb[:,:,i] = lbl_gray
            lbl = np.zeros((lbl_rgb.shape[0], lbl_rgb.shape[1]), dtype=np.uint8)

            for label_idx, (_, mask) in enumerate(self.label_mask_dict.items()):
                res = lbl_rgb == mask
                res = (label_idx + 1) * res.all(axis=2)
                lbl += res.astype(np.uint8)

            self.img_list.append(img)
            self.lbl_list.append(lbl)
            # Filenames for debugging
            self.file_list.append(data_name)

        if self.classes:
            self.__filter_classes()

    def __create_mask_dict(self, img_dims):
        with open(self.class_dict_file, newline='', encoding='utf-8') as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                if row[0] == 'name':
                    continue

                label = row[0]
                label_mask = np.zeros((img_dims[0], img_dims[1], 3), dtype=np.uint8)
                label_mask[:, :, 0] = np.uint8(row[1])
                label_mask[:, :, 1] = np.uint8(row[2])
                label_mask[:, :, 2] = np.uint8(row[3])

                self.label_mask_dict[label] = label_mask

    def __filter_classes(self):
        for e in self.lbl_list:
            initial_new_class_label = len(self.class_dict) + 5
            new_class_label = initial_new_class_label
            for l_class in self.classes:
                if l_class not in self.class_dict:
                    print(f'Class is not in the data: {l_class}')
                    return

                e[(e == self.class_dict[l_class])] = new_class_label
                new_class_label += 1

            e[(e < initial_new_class_label)] = new_class_label
            e -= initial_new_class_label

    @classmethod
    def pad_image_and_label(cls, img, lbl_rgba):

        """Crops square image from the original image crp_idx determines the crop area"""

        img_pad = Image.new(img.mode, (352, 352))
        img_pad.paste(img, (0, 0))

        lbl_pad = Image.new(lbl_rgba.mode, (352,352))
        lbl_pad.paste(lbl_rgba, (0,0))
        return (img_pad, lbl_pad)

    @staticmethod
    def normalize(data):
        """Normalizes data to the range [0, 1)"""
        return data / 256.

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.transform is not None:
            img = self.transform(self.img_list[idx])
        return img, self.lbl_list[idx].astype(np.int64)

def rfsegment_get_datasets_s256(data, load_train=True, load_test=True, num_classes=3):
    """
    Load the rfsegment dataset in 48x88x88 format which are composed of 3x352x352 images folded
    with a fold_ratio of 4.

    The dataset originally includes 33 keywords. A dataset is formed with 4 or 34 classes which
    includes 3, or 33 of the original keywords and the rest of the dataset is used to form the
    last class, i.e class of the others.

    The dataset is split into training+validation and test sets. 90:10 training+validation:test
    split is used by default.
    """
    (data_dir, args) = data

    classes = ['LTE', 'NR']

    transform = transforms.Compose([transforms.ToTensor(),
                                    ai8x.normalize(args=args),
                                    ai8x.fold(fold_ratio=4)])

    if load_train:
        train_dataset = RFSegment(root_dir=os.path.join(data_dir, 'RFSegment'), d_type='train',
                                      im_size=[256, 256], classes=classes,
                                      transform=transform)
    else:
        train_dataset = None

    if load_test:
        test_dataset = RFSegment(root_dir=os.path.join(data_dir, 'RFSegment'), d_type='test',
                                     im_size=[256, 256], classes=classes,
                                     transform=transform)

        if args.truncate_testset:
            test_dataset.img_list = test_dataset.img_list[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset

def rfsegment_get_datasets_s352(data, load_train=True, load_test=True, num_classes=2):
    """
    Load the rfsegment dataset in 48x88x88 format which are composed of 3x352x352 images folded
    with a fold_ratio of 4.

    The dataset originally includes 33 keywords. A dataset is formed with 4 or 34 classes which
    includes 3, or 33 of the original keywords and the rest of the dataset is used to form the
    last class, i.e class of the others.

    The dataset is split into training+validation and test sets. 90:10 training+validation:test
    split is used by default.
    """
    (data_dir, args) = data

    classes = ['LTE', 'NR']

    transform = transforms.Compose([transforms.ToTensor(),
                                    ai8x.normalize(args=args),
                                    ai8x.fold(fold_ratio=4)])

    if load_train:
        train_dataset = RFSegment(root_dir=os.path.join(data_dir, 'RFSegment'), d_type='train',
                                      im_size=[352, 352], classes=classes,
                                      transform=transform)
    else:
        train_dataset = None

    if load_test:
        test_dataset = RFSegment(root_dir=os.path.join(data_dir, 'RFSegment'), d_type='test',
                                     im_size=[352, 352], classes=classes,
                                     transform=transform)

        if args.truncate_testset:
            test_dataset.img_list = test_dataset.img_list[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


def rfsegment_get_datasets_s256_c3(data, load_train=True, load_test=True):
    """
    Load the rfsegment dataset for 3 classes in 48x64x64 images.
    """
    return rfsegment_get_datasets_s256(data, load_train, load_test, num_classes=3)

def rfsegment_get_datasets_s352_c3(data, load_train=True, load_test=True):
    """
    Load the rfsegment dataset for 3 classes in 48x88x88 images.
    """
    return rfsegment_get_datasets_s352(data, load_train, load_test, num_classes=3)


datasets = [
    {
        'name': 'RFSegment_s256_c2',  # 2 classes
        'input': (48, 64, 64),
        'output': (0, 1, 2),
        'weight': (1, 1, 1),
        'loader': rfsegment_get_datasets_s256_c3,
    },
    {
        'name': 'RFSegment_s352_c2',  # 2 classes
        'input': (48, 88, 88),
        'output': (0, 1, 2),
        'weight': (1, 1, 1),
        'loader': rfsegment_get_datasets_s352_c3,
        'fold_ratio': 4,
    }
]
