# Source: https://raw.githubusercontent.com/kenshohara/3D-ResNets-PyTorch/master/datasets/activitynet.py

import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import numpy as np
import nibabel as nib
from skimage.transform import rotate
import random
import math
import functools
import json
import copy
import cv2
from scipy.ndimage.filters import gaussian_filter
import scipy
from imgcrop import get_cropper
from config_h_r import parse_opts
config = parse_opts()

class ToTensor(object):
    """Convert ndarrays in sample to Tensors. Modifiy code from pytorch offical website"""

    def __call__(self, sample):

        return torch.from_numpy(sample).type(torch.FloatTensor)


def load_mask(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    mask = data > 0

    return mask


class MRI_h_r(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None):

        self.transform  = transforms.Compose([ToTensor()])
        # Get all subfolders
        self.subset = subset
        self.root = os.path.join(root_path, subset)
        self.image_folders = [(p,p[-10])
                  for o in os.listdir(self.root)
                  for p in os.listdir(os.path.join(self.root,o))
                  if p.endswith('mask.nii')]

        # collect patient level label
        label_dict = {}
        for o in os.listdir(self.root):
            label_dict[o] = 0
            for f in os.listdir(os.path.join(self.root, o)):
                if f.endswith('mask.nii') and f[-10] == '1':
                    label_dict[o] = 1
        self.label_dict = label_dict
        self.slice_dict = {}
        # constrcut a dict store each slice's label
        for (name, label) in self.image_folders:
            parts = name.split('_')
            study_num = parts[0]
            slice_num = parts[1]
            if study_num in self.slice_dict.keys():
                self.slice_dict[study_num][slice_num] = label
            else:
                temp_dict = {}
                temp_dict[slice_num] = label
                self.slice_dict[study_num] = temp_dict

        self.length = len(self.label_dict)
        self.all_keys = list(self.label_dict.keys())


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # initialize output tensor
        samples = torch.zeros([15,config.in_chanel, 256, 256])
        targets = torch.zeros([15,1]).type(torch.LongTensor)
        loss_masks = torch.zeros([15, 1])
        target_hs = torch.zeros([15, 1]).type(torch.LongTensor)
        target_hhs = torch.zeros([15, 1]).type(torch.LongTensor)


        study_num = self.all_keys[index]
        for slice_num in range(15):
            slice_num = str(slice_num)
            label = self.slice_dict[study_num][slice_num]
            # (img,label) = self.image_folders[index]
            # img_parts = img.split('_')
            running_instance = os.path.join(self.root,study_num,study_num+'_'+slice_num+'_')
            loss_mask = np.ones((1)).astype(np.float32)

            if os.path.exists(running_instance + str(label)+'_mask.nii'):
                mask = load_mask(running_instance + str(label)+'_mask.nii')
            else:
                print('Rrror: Not mask', running_instance)

            # Hierarchical training only see if there is tumor
            if label == '3':   # is it abnormal
                label_hh = '0'
            else:
                label_hh = '1'

            if label == '1' or label == '0':   # is there tumor
                label_h = '1'
            else:
                label_h = '0'

            # make it binary classification    is it recurrent tumor
            if label == '2' or label == '3':
                label = '0'


            array_3d = np.zeros([config.in_chanel,256,256]).astype(np.float32)

            if config.in_chanel ==5:
                sequences = ['apt.nii','T1.nii','Flair.nii','T2.nii','T1c.nii']
            elif config.in_chanel ==4:
                sequences = ['T1.nii','Flair.nii','T2.nii','T1c.nii']


            random_crop = get_cropper(patch_size=256, scale=[0.8, 1.2], rotate=[-180, 180],
                                      distort=0.05,
                                      flip=0.5, margin=0)

            list_sequences = []
            for sequence in sequences:
                img_path = running_instance+sequence
                img = nib.load(img_path)
                data = img.get_fdata().astype(np.float32)
                if config.apply_mask:
                    data =  np.multiply(data,mask)
                list_sequences.append(data)

            if self.subset == 'train':
                list_sequences = random_crop(list_sequences)[0]
            #else:
            #    list_sequences = random_crop_val(list_sequences)[0]

            for i, data in enumerate(list_sequences):
                data = data * 2 - 1
                array_3d[i, :, :] = data

            # label_one_hot = np.zeros((2))
            # label_one_hot[int(label)] = 1.0
            # target = self.transform(label_one_hot)

            label = np.array([float(label)]).astype(np.float32)
            target = self.transform(label).type(torch.LongTensor)

            label_h = np.array([float(label_h)]).astype(np.float32)
            target_h = self.transform(label_h).type(torch.LongTensor)

            label_hh = np.array([float(label_hh)]).astype(np.float32)
            target_hh = self.transform(label_hh).type(torch.LongTensor)

            sample = self.transform(array_3d)
            loss_mask = self.transform(loss_mask)

            # store patient level tesnors
            slice_num = int(slice_num)
            samples[slice_num,:,:,:] = sample
            targets[slice_num,:] = target
            loss_masks[slice_num,:] = loss_mask
            target_hs[slice_num,:] = target_h
            target_hhs[slice_num,:] = target_hh

        # patient level label
        label_p = np.array([float(self.label_dict[study_num])]).astype(np.float32)
        target_p = self.transform(label_p).type(torch.LongTensor)

        return samples, targets, loss_masks, target_hs, target_hhs, study_num, target_p

