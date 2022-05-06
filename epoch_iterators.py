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
# Date Created: 2018-XX-XX

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from utils.utils import *
import numpy as np
import torch
import pandas as pd

def one_hot_coding(config,true):
    true = true.detach().cpu().numpy()
    n_sample = true.shape[0]
    label_one_hot = np.zeros((n_sample,config.num_classes)).astype(np.float32)
    for i in range(n_sample):
        label_one_hot[i,true[i]] = 1.0
    return torch.from_numpy(label_one_hot).cuda()


def train_epoch(config, model, criterion, optimizer, device,
                data_loader, epoch, summary_writer=None):

    print('#'*60)
    print('Epoch {}. Starting with training phase.'.format(epoch+1))

    model.train()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loader.dataset)/config.batch_size))
    losses = np.zeros(steps_in_epoch, np.float32)
    accuracies = np.zeros(steps_in_epoch, np.float32)

    epoch_start_time = time.time()
    count_total = 0
    count_correct = 0
    for step, (clips, targets, loss_mask) in enumerate(data_loader):

        start_time = time.time()

        # Prepare for next iteration
        optimizer.zero_grad()

        # Move inputs to GPU memory
        clips = clips.cuda()
        targets = targets.cuda().view(-1)
        loss_mask = loss_mask.cuda().view(-1)



        # Feed-forward through the network
        logits = model.forward(clips)

        _, preds = torch.max(logits, 1)
        loss = torch.mean(criterion(logits, targets)*loss_mask)
        
        # #reverse one-hot
        # _, targets = torch.max(targets, 1)

        # Calculate accuracy
        correct = torch.sum(preds == targets.data)
        accuracy = correct.double() / clips.size()[0]
        count_total += clips.size()[0]
        count_correct += correct.double()

        # Calculate elapsed time for this step
        examples_per_second = config.batch_size/float(time.time() - start_time)

        # Back-propagation and optimization step
        loss.backward()
        optimizer.step()

        # Save statistics
        accuracies[step] = accuracy.item()
        losses[step] = loss.item()

        # Compute the global step, only for logging
        global_step = (epoch*steps_in_epoch) + step

        if step % config.print_frequency == 0:
            print("[{}] Epoch {}. Train Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                  "LR = {:.8f}, Accuracy = {:.3f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%A %H:%M"), epoch+1,
                    step, steps_in_epoch, examples_per_second,
                    current_learning_rate(optimizer), accuracies[step], losses[step]))

        if summary_writer and step % config.log_frequency == 0:
            summary_writer.add_scalar('train/loss', losses[step], global_step)
            summary_writer.add_scalar('train/accuracy', accuracies[step], global_step)
            summary_writer.add_scalar('train/examples_per_second', examples_per_second, global_step)
            summary_writer.add_scalar('train/learning_rate', current_learning_rate(optimizer), global_step)
            summary_writer.add_scalar('train/weight_decay', current_weight_decay(optimizer), global_step)


    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = np.mean(losses)
    epoch_avg_acc  = count_correct/count_total

    if summary_writer:
        summary_writer.add_scalar('train/epoch_avg_loss', epoch_avg_loss, epoch)
        summary_writer.add_scalar('train/epoch_avg_accuracy', epoch_avg_acc, epoch)

    return epoch_avg_loss, epoch_avg_acc, epoch_duration


####################################################################
####################################################################


def validation_epoch(config, model, criterion, device, data_loader, epoch, summary_writer=None):

    print('#'*60)
    print('Epoch {}. Starting with validation phase.'.format(epoch+1))

    model.eval()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loader.dataset)/config.batch_size))
    losses = np.zeros(steps_in_epoch, np.float32)
    accuracies = np.zeros(steps_in_epoch, np.float32)

    epoch_start_time = time.time()
    count_total = 0
    count_correct = 0
    for step, (clips, targets, loss_mask) in enumerate(data_loader):

        start_time = time.time()

        # Move inputs to GPU memory
        clips   = clips.cuda()
        targets = targets.cuda().view(-1)
        loss_mask = loss_mask.cuda().view(-1)
        #targets = targets.cuda()
        # Feed-forward through the network
        logits = model.forward(clips)

        _, preds = torch.max(logits, 1)
        #loss = criterion(logits, targets)
        loss = torch.mean(criterion(logits, targets) * loss_mask)

        # #reverse one-hot
        # _, targets = torch.max(targets, 1)

        # Calculate accuracy
        correct = torch.sum(preds == targets.data)
        accuracy = correct.double() / clips.size()[0]
        count_total += clips.size()[0]
        count_correct += correct.double()
        # Calculate elapsed time for this step
        examples_per_second = config.batch_size/(float(time.time() - start_time)+10e-8)

        # Save statistics
        accuracies[step] = accuracy.item()
        losses[step] = loss.item()

        if step % config.print_frequency == 0:
            print("[{}] Epoch {}. Validation Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.3f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%A %H:%M"), epoch+1,
                    step, steps_in_epoch, examples_per_second,
                    accuracies[step], losses[step]))
        if step == 0:
            cum_targets = targets.detach().cpu()
            cum_scores = torch.nn.functional.softmax(logits.detach().cpu(),dim=1)
        else:
            cum_targets = torch.cat((cum_targets,targets.detach().cpu()),0)
            cum_scores = torch.cat((cum_scores,  torch.nn.functional.softmax(logits.detach().cpu(),dim=1)), 0)

    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = np.mean(losses)
    epoch_avg_acc  = count_correct/count_total

    avg_auc_score,accuracy, sens, spec, optimal_threshold = plot_roc(cum_targets,cum_scores)

    if summary_writer:
        summary_writer.add_scalar('validation/epoch_avg_loss', epoch_avg_loss, epoch)
        summary_writer.add_scalar('validation/epoch_avg_accuracy', epoch_avg_acc, epoch)
        summary_writer.add_scalar('validation/epoch_true_accuracy', accuracy, epoch)
        summary_writer.add_scalar('validation/epoch_true_auc', avg_auc_score, epoch)
        summary_writer.add_scalar('validation/epoch_true_sens', sens, epoch)
        summary_writer.add_scalar('validation/epoch_true_spec', spec, epoch)
        summary_writer.add_scalar('validation/epoch_optimal_threshold', optimal_threshold, epoch)

    return epoch_avg_loss, accuracy, epoch_duration


def plot_roc(known_scores, unknown_scores):
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
    y_true = known_scores.detach().cpu().numpy()
    y_score = unknown_scores.detach().cpu().numpy()[:,1]

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    avg_auc_score = roc_auc_score(y_true, y_score,average='weighted')

    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(thresholds, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    #print('AUC {:.03f}'.format(auc_score))
    optimal_threshold = roc_t['threshold'].values[0]

    y_pred = np.zeros_like(y_true)
    y_pred[y_score>optimal_threshold] = 1.0

    correct = np.sum(y_pred == y_true)
    accuracy = correct/y_true.shape[0]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)

    return avg_auc_score,accuracy, sens, spec, optimal_threshold