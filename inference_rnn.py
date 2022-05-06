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
import pandas as pd
import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, ToTensor, \
    CenterCrop
from transforms.temporal_transforms import TemporalRandomCrop
from transforms.target_transforms import ClassLabel
import pickle
from epoch_iterators_h_r import train_epoch, validation_epoch
from utils.utils import *
import utils.mean_values
import factory.data_factory as data_factory
import factory.model_factory as model_factory
from config_h_r import parse_opts
from rnn_p import LSTMTagger
import multiprocessing


def lambda_rule(epoch):
    '''
          for 'linear', we keep the same learning rate for the first <opt.niter> epochs
          and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    '''
    oepoch_count = 0
    niter = 50
    niter_decay = 50
    if epoch < 5:
        # first several epochs gives 0.1*initial lr
        lr_l = 0.1
    else:
        # Then return to original scheduler
        lr_l = 1.0 - max(0, epoch + oepoch_count - niter) / float(niter_decay + 1)
    return lr_l


def main():
    ####################################################################
    ####################################################################
    # Configuration and logging
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = parse_opts()
    config.video_path = '/mass/pengfei/recurrece_prediction_0619/post_data_2020_0219_2d_all_slice_all_label'
    config.save_dir = '/mass/pengfei/recurrece_prediction_0619/output/inference/debug_analysis'
    config = prepare_output_dirs(config)
    config = init_cropping_scales(config)
    config = set_lr_scheduling_policy(config)

    config.image_mean = utils.mean_values.get_mean(config.norm_value, config.dataset)
    config.image_std = utils.mean_values.get_std(config.norm_value)
    rnn_checkpoint = '/mass/pengfei/recurrece_prediction_0619/output/plot_roc_and_analysis/h_ls_r/w_apt/1.pth'
    config.manual_seed = 666
    save_raw = 1  # save the raw data that can plot ROC curve
    raw_save_path = '/mass/pengfei/recurrece_prediction_0619/output/plot_roc_and_analysis/h_ls_r/w_apt/dict_1.p'

    ####################################################################
    ####################################################################
    # Initialize model
    device = torch.device(config.device)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed_all(config.manual_seed)
    np.random.seed(config.manual_seed)

    # Returns the network instance (I3D, 3D-ResNet etc.)
    # Note: this also restores the weights and optionally replaces final layer
    model, parameters = model_factory.get_model(config)
    # LSTM
    model_r = LSTMTagger(512, 2)
    if 'cuda' in config.device:
        print('Moving RNN to CUDA device...')
        # Move model to the GPU
        model_r = model_r.cuda()
        model_r = nn.DataParallel(model_r, device_ids=None)
        model_r.load_state_dict(torch.load(rnn_checkpoint)['state_dict'])
        print('Loaded RNN checkpoint : ', rnn_checkpoint)

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
    # weights = [1, (413+996)/751]
    # class_weights = torch.FloatTensor(weights).cuda()
    # weights = [89/210, 40/210, 81/210]
    # class_weights = torch.FloatTensor(weights).cuda()
    # criterion = nn.CrossEntropyLoss(class_weights,reduction='none').to(device)
    # reduction='none'
    weights = [1742 / 418, 1]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    weights = [1164 / 996, 1]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion_h = nn.CrossEntropyLoss(weight=class_weights).to(device)
    weights = [1, 1409 / 751]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion_hh = nn.CrossEntropyLoss(weight=class_weights).to(device)

    weights = [86 / 58, 1]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion_r = nn.CrossEntropyLoss(weight=class_weights).to(device)



    ####################################################################
    ####################################################################
    def plot_roc(known_scores, unknown_scores, cum_study_number=None):
        from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
        y_true = known_scores.detach().cpu().numpy()
        y_score = unknown_scores.detach().cpu().numpy()[:, 1]

        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        avg_auc_score = roc_auc_score(y_true, y_score, average='weighted')

        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(thresholds, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
        # print('AUC {:.03f}'.format(auc_score))
        optimal_threshold = roc_t['threshold'].values[0]

        y_pred = np.zeros_like(y_true)
        y_pred[y_score > optimal_threshold] = 1.0

        correct = np.sum(y_pred == y_true)
        accuracy = correct / y_true.shape[0]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)

        return avg_auc_score, accuracy, sens, spec, optimal_threshold

    def evaluate(config, model, model_r, criterion, criterion_h, criterion_hh, criterion_r, device, data_loader):

        print('#' * 60)

        model.eval()

        # Epoch statistics
        steps_in_epoch = int(np.ceil(len(data_loader.dataset) / config.batch_size))
        losses = 0
        accuracies = np.zeros(steps_in_epoch, np.float32)

        epoch_start_time = time.time()
        count_total = 0
        count_correct = 0
        model.eval()
        for step, (clips_, targets_, loss_mask_, targets_h_, targets_hh_, study_number_, targets_p) in enumerate(
                data_loader):
            start_time = time.time()
            features = torch.zeros([clips_.shape[0], 15, 512]).cuda()
            # slice level
            with torch.no_grad():
                for batch in range(clips_.shape[0]):
                    # Move inputs to GPU memory
                    clips = clips_[batch].cuda()
                    targets = targets_[batch].cuda().view(-1)
                    loss_mask = loss_mask_[batch].cuda().view(-1)
                    targets_h = targets_h_[batch].cuda().view(-1)
                    targets_hh = targets_hh_[batch].cuda().view(-1)

                    # Feed-forward through the network
                    logits, logits_h, logits_hh, feature = model.forward(clips)
                    features[batch] = feature

                    _, preds = torch.max(logits, 1)
                    loss = criterion(logits, targets)

                    # H
                    if config.Hierarchical:
                        loss_hh = criterion_hh(logits_hh, targets_hh)
                        loss_h = criterion_h(logits_h, targets_h)
                        loss = (loss + loss_h + loss_hh) / 3.0

                    if step == 0 and batch == 0:
                        cum_study_number = [study_number_[batch]] * 15

                        cum_targets = targets.detach().cpu()
                        cum_scores = torch.nn.functional.softmax(logits.detach().cpu(), dim=1)

                        cum_targets_h = targets_h.detach().cpu()
                        cum_scores_h = torch.nn.functional.softmax(logits_h.detach().cpu(), dim=1)

                        cum_targets_hh = targets_hh.detach().cpu()
                        cum_scores_hh = torch.nn.functional.softmax(logits_hh.detach().cpu(), dim=1)
                    else:
                        cum_study_number += [study_number_[batch]] * 15

                        cum_targets = torch.cat((cum_targets, targets.detach().cpu()), 0)
                        cum_scores = torch.cat((cum_scores, torch.nn.functional.softmax(logits.detach().cpu(), dim=1)),
                                               0)

                        cum_targets_h = torch.cat((cum_targets_h, targets_h.detach().cpu()), 0)
                        cum_scores_h = torch.cat(
                            (cum_scores_h, torch.nn.functional.softmax(logits_h.detach().cpu(), dim=1)), 0)

                        cum_targets_hh = torch.cat((cum_targets_hh, targets_hh.detach().cpu()), 0)
                        cum_scores_hh = torch.cat(
                            (cum_scores_hh, torch.nn.functional.softmax(logits_hh.detach().cpu(), dim=1)), 0)
            # slice level end

            # patient level
            targets_p = targets_p.cuda().view(-1)
            # forward rnn
            model_r.eval()
            logits_p = model_r(features)
            _, preds = torch.max(logits_p, 1)
            loss_p = criterion_r(logits_p, targets_p)

            # Calculate accuracy
            correct = torch.sum(preds == targets_p.data)
            accuracy = correct.double() / clips.size()[0]
            count_total += clips.size()[0]
            count_correct += correct.double()
            # Calculate elapsed time for this step
            examples_per_second = config.batch_size / (float(time.time() - start_time) + 10e-8)

            # Save statistics
            accuracies[step] = accuracy.item()
            losses += loss_p.item()

            if step % config.print_frequency == 0:
                print("[{}] Epoch {}. Validation Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.3f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%A %H:%M"), 0 + 1,
                    step, steps_in_epoch, examples_per_second,
                    accuracies[step], loss_p.item()))
            if step == 0:
                cum_targets_p = targets_p.detach().cpu()
                cum_scores_p = torch.nn.functional.softmax(logits_p.detach().cpu(), dim=1)

            else:
                cum_targets_p = torch.cat((cum_targets_p, targets_p.detach().cpu()), 0)
                cum_scores_p = torch.cat((cum_scores_p, torch.nn.functional.softmax(logits_p.detach().cpu(), dim=1)), 0)

        # Epoch statistics
        epoch_duration = float(time.time() - epoch_start_time)
        epoch_avg_loss = losses / (step + 1)
        epoch_avg_acc = count_correct / count_total
        cum_study_number = cum_study_number[::15]
        avg_auc_score_p, accuracy_p, sens_p, spec_p, optimal_threshold_p = plot_roc(cum_targets_p, cum_scores_p)
        if save_raw:
            save_dict = {'cum_targets': cum_targets_p.detach().cpu().numpy(),
                         'cum_scores': cum_scores_p.detach().cpu().numpy()[:, 1],
                         'cum_study_number': cum_study_number,
                         'auc': avg_auc_score_p,
                         'checkpoint': config.checkpoint_path}
            pickle.dump(save_dict, open(raw_save_path, 'wb'))
        print('auc: ', avg_auc_score_p, 'acc: ', accuracy_p, 'sens: ', sens_p, 'spec: ', spec_p, 'threshold: ', optimal_threshold_p)

    # Perform evaluetion
    evaluate(
        config=config,
        model=model,
        model_r=model_r,
        criterion=criterion,
        criterion_h=criterion_h,
        criterion_hh=criterion_hh,
        criterion_r=criterion_r,
        device=device,
        data_loader=data_loaders['validation'],
    )











if __name__ == '__main__':
    main()