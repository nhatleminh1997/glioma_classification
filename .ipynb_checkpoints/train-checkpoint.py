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
import torch
import torch.nn as nn

from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, ToTensor, \
    CenterCrop
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel

from epoch_iterators import train_epoch, validation_epoch
from utils.utils import *
import utils.mean_values
import factory.data_factory as data_factory
import factory.model_factory as model_factory
from config import parse_opts

def lambda_rule(epoch):
    '''
          for 'linear', we keep the same learning rate for the first <opt.niter> epochs
          and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    '''
    oepoch_count = 0
    niter = 50
    niter_decay = 50
    if epoch < 1:
        #first several epochs gives 0.1*initial lr
        lr_l = 0.1 
    else:
        #Then return to original scheduler
        lr_l = 1.0 - max(0, epoch + oepoch_count - niter) / float(niter_decay + 1)
    return lr_l

def main():
    ####################################################################
    ####################################################################
    # Configuration and logging
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    config = parse_opts()
    config = prepare_output_dirs(config)
    config = init_cropping_scales(config)
    config = set_lr_scheduling_policy(config)

    config.image_mean = utils.mean_values.get_mean(config.norm_value, config.dataset)
    config.image_std = utils.mean_values.get_std(config.norm_value)

    print_config(config)
    write_config(config, os.path.join(config.save_dir, 'config.json'))

    # TensorboardX summary writer
    if not config.no_tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(config.save_dir)
    else:
        writer = None

    ####################################################################
    ####################################################################
    # Initialize model
    device = torch.device(config.device)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed_all(config.manual_seed)
    
    # Returns the network instance (I3D, 3D-ResNet etc.)
    # Note: this also restores the weights and optionally replaces final layer
    model, parameters = model_factory.get_model(config)

    print('#' * 60)
    if config.model == 'i3d':
        # param_names = [p[name] for p in list(parameters)]
        # print('Parameters to train:')
        # print(param_names)
        # print(len(parameters))
        print('#' * 60)

    ####################################################################
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
        'spatial': Compose([MultiScaleRandomCrop(config.scales, config.spatial_size),
                            RandomHorizontalFlip(),
                            ToTensor(config.norm_value),
                            norm_method]),
        'temporal': TemporalRandomCrop(config.sample_duration),
        'target': ClassLabel()
    }

    # print('WARNING: setting train transforms for dataset statistics')
    # train_transforms = {
    #     'spatial':  Compose([ToTensor(1.0)]),
    #     'temporal': TemporalRandomCrop(64),
    #     'target':   ClassLabel()
    # }

    validation_transforms = {
        'spatial': Compose([CenterCrop(config.spatial_size),
                            ToTensor(config.norm_value),
                            norm_method]),
        'temporal': TemporalRandomCrop(config.sample_duration),
        'target': ClassLabel()
    }

    ####################################################################
    ####################################################################
    # Setup of data pipeline

    data_loaders = data_factory.get_data_loaders(config, train_transforms, validation_transforms)
    phases = ['train', 'validation'] if 'validation' in data_loaders else ['train']
    print('#' * 60)

    ####################################################################
    ####################################################################
    # Optimizer and loss initialization
    weights = [797/367, 1]
    class_weights = torch.FloatTensor(weights).cuda()
#     weights = [89/210, 40/210, 81/210]
#     class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(class_weights).to(device)
    #criterion = nn.CrossEntropyLoss().to(device)
    optimizer = get_optimizer(config, parameters)

    '''
       for 'linear', we keep the same learning rate for the first <opt.niter> epochs
       and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    '''

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


    ####################################################################
    ####################################################################

    # Keep track of best validation accuracy
    val_acc_history = []
    best_val_acc = 0.0

    for epoch in range(config.start_epoch, config.num_epochs + 1):

        # First 'training' phase, then 'validation' phase
        for phase in phases:

            if phase == 'train':

                # Perform one training epoch
                train_loss, train_acc, train_duration = train_epoch(
                    config=config,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    data_loader=data_loaders['train'],
                    epoch=epoch,
                    summary_writer=writer
                )

            elif phase == 'validation':

                # Perform one training epoch
                val_loss, val_acc, val_duration = validation_epoch(
                    config=config,
                    model=model,
                    criterion=criterion,
                    device=device,
                    data_loader=data_loaders['validation'],
                    epoch=epoch,
                    summary_writer=writer
                )

                val_acc_history.append(val_acc)

        # Update learning rate

        # if config.lr_scheduler == 'plateau':
        #     scheduler.step(val_loss)
        # else:
        scheduler.step()

        print('#' * 60)
        print('EPOCH {} SUMMARY'.format(epoch + 1))
        print('Training Phase.')
        print('  Total Duration:              {} minutes'.format(int(np.ceil(train_duration / 60))))
        print('  Average Train Loss:          {:.3f}'.format(train_loss))
        print('  Average Train Accuracy:      {:.3f}'.format(train_acc))

        if 'validation' in phases:
            print('Validation Phase.')
            print('  Total Duration:              {} minutes'.format(int(np.ceil(val_duration / 60))))
            print('  Average Validation Loss:     {:.3f}'.format(val_loss))
            print('  Average Validation Accuracy: {:.3f}'.format(val_acc))

        if 'validation' in phases and val_acc > best_val_acc:
            checkpoint_path = os.path.join(config.checkpoint_dir, str(val_acc.item()) + '_'+ str(epoch + 1) +'_save_best.pth')
            save_checkpoint(checkpoint_path, epoch, model.state_dict(), optimizer.state_dict())
            print('Found new best validation accuracy: {:.3f}'.format(val_acc))
            print('Model checkpoint (best) written to:     {}'.format(checkpoint_path))
            best_val_acc = val_acc

        # Model saving
        if epoch % config.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, 'save_{:03d}.pth'.format(epoch + 1))
            save_checkpoint(checkpoint_path, epoch, model.state_dict(), optimizer.state_dict())
            print('Model checkpoint (periodic) written to: {}'.format(checkpoint_path))
            cleanup_checkpoint_dir(config)  # remove old checkpoint files


    # Dump all TensorBoard logs to disk for external processing
    writer.export_scalars_to_json(os.path.join(config.save_dir, 'all_scalars.json'))
    writer.close()

    print('Finished training.')


if __name__ == '__main__':
    main()