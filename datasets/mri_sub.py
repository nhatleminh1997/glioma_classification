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
from config import parse_opts
config = parse_opts()

class ToTensor(object):
    """Convert ndarrays in sample to Tensors. Modifiy code from pytorch offical website"""

    def __call__(self, sample):

        return  torch.from_numpy(sample).type(torch.FloatTensor)


def load_mask(path):
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    mask = data > 0

    return mask


class MRI_sub(data.Dataset):
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
        self.image_folders = [p
                for o in os.listdir(self.root)
                for p in os.listdir(os.path.join(self.root,o))
                if p.endswith('mask.nii')]

        labels = [int(p[-10])
                for o in os.listdir(self.root)
                 for p in os.listdir(os.path.join(self.root, o))
                if p.endswith('mask.nii')]
        self.targets = np.array(labels, dtype=np.int64)
        self.labels =torch.from_numpy(np.array(labels, dtype=np.int64))

        self.length = len(self.image_folders)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img = self.image_folders[index]
        label = self.targets[index]
        
        img_parts = img.split('_')
        running_instance = os.path.join(self.root,img_parts[0],img_parts[0]+'_'+img_parts[1]+'_')
       

        if os.path.exists(running_instance + str(label)+'_mask.nii'):
            mask = load_mask(running_instance + str(label)+'_mask.nii')
        else:
            print('Rrror: Not mask', running_instance)
        
        array_3d = np.zeros([config.in_chanel,256,256]).astype(np.float32)
        
        
        sequences = ['apt.nii','T1.nii','Flair.nii','T2.nii','T1c.nii']

            
        random_crop = get_cropper(patch_size=256, scale=[0.8, 1.2], rotate=[-45, 45],
                                  distort=0.05,
                                  flip=0.5, margin=0)
        
        random_crop_val = get_cropper(patch_size=256, scale=[1.0, 1.1], rotate=[-5, 5],
                                  distort=0.05,
                                  flip=0.5, margin=0)
        
        list_sequences = []
        for sequence in sequences:
            img_path = running_instance+sequence
            img = nib.load(img_path)
            data = img.get_fdata().astype(np.float32)
            #data =  np.multiply(data,mask)
            list_sequences.append(data)
           
            
        if self.subset == 'train':
            list_sequences = random_crop(list_sequences)[0]
        #else:
        #    list_sequences = random_crop_val(list_sequences)[0]

        #for i, data in enumerate(list_sequences):
        #    data = data * 2 - 1
        
        #apt
        array_3d[0, :, :] = (list_sequences[0]*2-1)
        # t1c -t1
        array_3d[1, :, :] = (list_sequences[4]-list_sequences[1])
        # Flair -t2
        array_3d[2, :, :] = (list_sequences[2]-list_sequences[3])
        
        #label = np.array([float(label)]).astype(np.float32)
        label_one_hot = np.zeros((2))
        label_one_hot[label] = 1.0
        target = self.transform(label_one_hot) #.type(torch.LongTensor)
        sample = self.transform(array_3d)

        return sample, target

