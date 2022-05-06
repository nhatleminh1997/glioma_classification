# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-03-01

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime

import torch.nn as nn

from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, ToTensor, CenterCrop
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel

from epoch_iterators import train_epoch, validation_epoch
from utils.utils import *
import utils.mean_values
import factory.data_factory as data_factory
import factory.model_factory as model_factory
from config import parse_opts


def main():
    ####################################################################
    ####################################################################
    # Configuration and logging

    config = parse_opts()
    config = prepare_output_dirs(config)
    config = init_cropping_scales(config)
    config = set_lr_scheduling_policy(config)

    config.image_mean = utils.mean_values.get_mean(config.norm_value, config.dataset)
    config.image_std = utils.mean_values.get_std(config.norm_value)

    print_config(config)
    write_config(config, os.path.join(config.save_dir, 'config.json'))


    ####################################################################
    ####################################################################
    # Initialize model

    device = torch.device(config.device)

    ####################################################################
    # Setup of data transformations

    if config.no_dataset_mean and config.no_dataset_std:
        # Just zero-center and scale to unit std
        print('Data normalization: no dataset mean, no dataset std')
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not config.no_dataset_mean and config.no_dataset_std:
        # Subtract dataset mean and scale to unit std
        print('Data normalization: use dataset mean, no dataset std')
        norm_method = Normalize(config.image_mean, [1, 1, 1])
    else:
        # Subtract dataset mean and scale to dataset std
        print('Data normalization: use dataset mean, use dataset std')
        norm_method = Normalize(config.image_mean, config.image_std)

    train_transforms = {
        'spatial':  Compose([MultiScaleRandomCrop(config.scales, config.spatial_size),
                             RandomHorizontalFlip(),
                             ToTensor(config.norm_value),
                             norm_method]),
        'temporal': TemporalRandomCrop(config.sample_duration),
        'target':   ClassLabel()
    }

    # print('WARNING: setting train transforms for dataset statistics')
    # train_transforms = {
    #     'spatial':  Compose([ToTensor(1.0)]),
    #     'temporal': TemporalRandomCrop(64),
    #     'target':   ClassLabel()
    # }

    validation_transforms = {
        'spatial':  Compose([CenterCrop(config.spatial_size),
                             ToTensor(config.norm_value),
                             norm_method]),
        'temporal': TemporalRandomCrop(config.sample_duration),
        'target':   ClassLabel()
    }

    ####################################################################
    ####################################################################
    # Setup of data pipeline

    data_loaders = data_factory.get_data_loaders(config, train_transforms, validation_transforms)
    phases = ['train']
    print('#'*60)

    ####################################################################
    ####################################################################
    #
    ####################################################################
    ####################################################################
    for phase in phases:
        if phase == 'train':
            data_loader = data_loaders['train']
            mean = 0.
            meansq = 0.
            for step, (clips, targets) in enumerate(data_loader):
                clips = clips[:,4,:,:,:].cuda()
                mean = clips.mean()
                meansq = (clips ** 2).mean()
            std = torch.sqrt(meansq - mean ** 2)
            print("mean: " + str(mean))
            print("std: " + str(std))
            print()

    print('Finished Measure')

if __name__ == '__main__':
    main()