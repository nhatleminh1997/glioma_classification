from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import torch

from transforms.spatial_transforms import Normalize
from torch.utils.data import DataLoader, Sampler

from datasets.kinetics import Kinetics
from datasets.activitynet import ActivityNet
from datasets.ucf101 import UCF101
from datasets.blender import BlenderSyntheticDataset
from datasets.mri import MRI
from datasets.mri_sub import MRI_sub
from datasets.mri_h import MRI_h
from datasets.mri_h_r import MRI_h_r

##########################################################################################
##########################################################################################

def get_training_set(config, spatial_transform, temporal_transform, target_transform):

    assert config.dataset in ['kinetics', 'activitynet', 'ucf101', 'blender','MRI','MRI_sub','MRI_H', 'MRI_H_R']

    if config.dataset == 'MRI':

        training_data = MRI(
            config.video_path,
            'train',
            spatial_transform=None,
            temporal_transform=None,
            target_transform=None)
        
    elif config.dataset == 'MRI_sub':

        training_data = MRI_sub(
            config.video_path,
            'train',
            spatial_transform=None,
            temporal_transform=None,
            target_transform=None)

    elif config.dataset == 'MRI_H':

        training_data = MRI_h(
            config.video_path,
            'train',
            spatial_transform=None,
            temporal_transform=None,
            target_transform=None)

    elif config.dataset == 'MRI_H_R':

        training_data = MRI_h_r(
            config.video_path,
            'train',
            spatial_transform=None,
            temporal_transform=None,
            target_transform=None)


    return training_data


##########################################################################################
##########################################################################################

def get_validation_set(config, spatial_transform, temporal_transform, target_transform):

    assert config.dataset in ['MRI','kinetics', 'activitynet', 'ucf101', 'blender','MRI_sub', 'MRI_H', 'MRI_H_R']

    # Disable evaluation
    if config.no_eval:
        return None

    if config.dataset == 'MRI':

        validation_data = MRI(
            config.video_path,
            'validation',
            spatial_transform=None,
            temporal_transform=None,
            target_transform=None)
    
    elif config.dataset == 'MRI_sub':

        validation_data = MRI_sub(
            config.video_path,
            'validation',
            spatial_transform=None,
            temporal_transform=None,
            target_transform=None)

    elif config.dataset == 'MRI_H':

        validation_data = MRI_h(
            config.video_path,
            'validation',
            spatial_transform=None,
            temporal_transform=None,
            target_transform=None)

    elif config.dataset == 'MRI_H_R':

        validation_data = MRI_h_r(
            config.video_path,
            'validation',
            spatial_transform=None,
            temporal_transform=None,
            target_transform=None)

    return validation_data

##########################################################################################
##########################################################################################

def get_test_set(config, spatial_transform, temporal_transform, target_transform):

    assert config.dataset in ['kinetics', 'activitynet', 'ucf101', 'blender']
    assert config.test_subset in ['val', 'test']

    if config.test_subset == 'val':
        subset = 'validation'
    elif config.test_subset == 'test':
        subset = 'testing'

    if config.dataset == 'kinetics':

        test_data = Kinetics(
            config.video_path,
            config.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    elif config.dataset == 'activitynet':

        test_data = ActivityNet(
            config.video_path,
            config.annotation_path,
            subset,
            True,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    elif config.dataset == 'ucf101':

        test_data = UCF101(
            config.video_path,
            config.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    return test_data


##########################################################################################
##########################################################################################

def get_normalization_method(config):
    if config.no_mean_norm and not config.std_norm:
        return Normalize([0, 0, 0], [1, 1, 1])
    elif not config.std_norm:
        return Normalize(config.mean, [1, 1, 1])
    else:
        return Normalize(config.mean, config.std)

##########################################################################################
##########################################################################################

class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np

        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)



def get_data_loaders(config, train_transforms, validation_transforms=None):

    print('[{}] Preparing datasets...'.format(datetime.now().strftime("%A %H:%M")))

    data_loaders = dict()

    # Define the data pipeline
    
    dataset_train = get_training_set(
        config, train_transforms['spatial'],
        train_transforms['temporal'], train_transforms['target'])



    #train_sampler = StratifiedSampler(dataset_train.labels, config.batch_size)
    data_loaders['train'] = DataLoader(
        dataset_train, config.batch_size,shuffle=True,
        num_workers=config.num_workers, pin_memory=True)

    print('Found {} training examples'.format(len(dataset_train)))

    if not config.no_eval and validation_transforms:

        dataset_validation = get_validation_set(
            config, validation_transforms['spatial'],
            validation_transforms['temporal'], validation_transforms['target'])

        print('Found {} validation examples'.format(len(dataset_validation)))

        data_loaders['validation'] = DataLoader(
            dataset_validation, config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=True)

    return data_loaders