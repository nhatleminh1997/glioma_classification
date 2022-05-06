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


def train_epoch(config, model, model_r, criterion, criterion_h, criterion_hh, criterion_r, optimizer, optimizer_r,device,
                data_loader, epoch, summary_writer=None):

    print('#'*60)
    print('Epoch {}. Starting with training phase.'.format(epoch+1))

    model.train()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loader.dataset)/config.batch_size))
    losses = 0
    accuracies = np.zeros(steps_in_epoch, np.float32)

    epoch_start_time = time.time()
    count_total = 0
    count_correct = 0
    model.eval()
    for step, (clips_, targets_, loss_mask_, targets_h_, targets_hh_, study_number_, targets_p) in enumerate(data_loader):
        start_time = time.time()
        features = torch.zeros([clips_.shape[0],15,512]).cuda()
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
                    loss = (loss + loss_h + loss_hh)/3.0
        # slice level end

        #patient level
        model_r.train()
        targets_p = targets_p.cuda().view(-1)
        # Prepare for next iteration
        optimizer_r.zero_grad()
        # forward rnn
        logits_p = model_r(features)
        _, preds = torch.max(logits_p, 1)
        loss_p = criterion_r(logits_p, targets_p)

        # Calculate accuracy
        correct = torch.sum(preds == targets_p.data)
        accuracy = correct.double() / clips.size()[0]
        count_total += clips.size()[0]
        count_correct += correct.double()

        # Calculate elapsed time for this step
        examples_per_second = config.batch_size/float(time.time() - start_time)

        # Back-propagation and optimization step
        loss_p.backward()
        optimizer_r.step()

        # Save statistics
        accuracies[step] = accuracy.item()
        losses += loss_p.item()

        # Compute the global step, only for logging
        global_step = (epoch*steps_in_epoch) + step


        if step % config.print_frequency == 0 and batch:
            print("[{}] Epoch {}. Train Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                  "LR = {:.8f}, Accuracy = {:.3f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%A %H:%M"), epoch+1,
                    step, steps_in_epoch, examples_per_second,
                    current_learning_rate(optimizer_r), accuracies[step], loss_p.item()))

        if summary_writer and step % config.log_frequency == 0:
            summary_writer.add_scalar('train/loss', loss_p.item(), global_step)
            summary_writer.add_scalar('train/accuracy', accuracies[step], global_step)
            summary_writer.add_scalar('train/examples_per_second', examples_per_second, global_step)
            summary_writer.add_scalar('train/learning_rate', current_learning_rate(optimizer_r), global_step)
            summary_writer.add_scalar('train/weight_decay', current_weight_decay(optimizer_r), global_step)


    # Epoch statistics
    epoch_duration = float(time.time() - epoch_start_time)
    epoch_avg_loss = losses/(step+1)
    epoch_avg_acc = count_correct/count_total

    if summary_writer:
        summary_writer.add_scalar('train/epoch_avg_loss', epoch_avg_loss, epoch)
        summary_writer.add_scalar('train/epoch_avg_accuracy', epoch_avg_acc, epoch)

    return epoch_avg_loss, epoch_avg_acc, epoch_duration


####################################################################
####################################################################


def validation_epoch(config, model, model_r, criterion, criterion_h, criterion_hh, criterion_r, device, data_loader, epoch, summary_writer=None):

    print('#'*60)
    print('Epoch {}. Starting with validation phase.'.format(epoch+1))

    model.eval()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loader.dataset)/config.batch_size))
    losses = 0
    accuracies = np.zeros(steps_in_epoch, np.float32)

    epoch_start_time = time.time()
    count_total = 0
    count_correct = 0
    model.eval()
    for step, (clips_, targets_, loss_mask_, targets_h_, targets_hh_, study_number_, targets_p) in enumerate(data_loader):
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

                if step == 0 and batch == 0 :
                    cum_study_number = [study_number_[batch]]*15

                    cum_targets = targets.detach().cpu()
                    cum_scores = torch.nn.functional.softmax(logits.detach().cpu(), dim=1)

                    cum_targets_h = targets_h.detach().cpu()
                    cum_scores_h = torch.nn.functional.softmax(logits_h.detach().cpu(), dim=1)

                    cum_targets_hh = targets_hh.detach().cpu()
                    cum_scores_hh = torch.nn.functional.softmax(logits_hh.detach().cpu(), dim=1)
                else:
                    cum_study_number += [study_number_[batch]]*15

                    cum_targets = torch.cat((cum_targets, targets.detach().cpu()), 0)
                    cum_scores = torch.cat((cum_scores, torch.nn.functional.softmax(logits.detach().cpu(), dim=1)), 0)

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
        examples_per_second = config.batch_size/(float(time.time() - start_time)+10e-8)

        # Save statistics
        accuracies[step] = accuracy.item()
        losses += loss_p.item()

        if step % config.print_frequency == 0:
            print("[{}] Epoch {}. Validation Step {:04d}/{:04d}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.3f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%A %H:%M"), epoch+1,
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
    epoch_avg_loss = losses/(step+1)
    epoch_avg_acc  = count_correct/count_total

    avg_auc_score,accuracy, sens, spec, optimal_threshold, _, _, _ = plot_roc(cum_targets, cum_scores, cum_study_number)

    avg_auc_score_h, accuracy_h, sens_h, spec_h, optimal_threshold_h = plot_roc(cum_targets_h, cum_scores_h)

    avg_auc_score_hh, accuracy_hh, sens_hh, spec_hh, optimal_threshold_hh = plot_roc(cum_targets_hh, cum_scores_hh)

    avg_auc_score_p, accuracy_p, sens_p, spec_p, optimal_threshold_p = plot_roc(cum_targets_p, cum_scores_p)

    if summary_writer:
        summary_writer.add_scalar('validation/epoch_avg_loss', epoch_avg_loss, epoch)
        summary_writer.add_scalar('validation/epoch_avg_accuracy', epoch_avg_acc, epoch)

        summary_writer.add_scalar('validation/slice_level/epoch_true_accuracy', accuracy, epoch)
        summary_writer.add_scalar('validation/slice_level/epoch_true_auc', avg_auc_score, epoch)
        summary_writer.add_scalar('validation/slice_level/epoch_true_sens', sens, epoch)
        summary_writer.add_scalar('validation/slice_level/epoch_true_spec', spec, epoch)
        summary_writer.add_scalar('validation/slice_level/epoch_optimal_threshold', optimal_threshold, epoch)

        summary_writer.add_scalar('validation/patient_level/epoch_true_accuracy_p',accuracy_p, epoch)
        summary_writer.add_scalar('validation/patient_level/epoch_true_auc_p', avg_auc_score_p, epoch)
        summary_writer.add_scalar('validation/patient_level/epoch_true_sens_p', sens_p, epoch)
        summary_writer.add_scalar('validation/patient_level/epoch_true_spec_p', spec_p, epoch)
        summary_writer.add_scalar('validation/patient_level/epoch_optimal_threshold', optimal_threshold_p, epoch)

        summary_writer.add_scalar('validation/h/epoch_true_accuracy_h', accuracy_h, epoch)
        summary_writer.add_scalar('validation/h/epoch_true_auc_h', avg_auc_score_h, epoch)

        summary_writer.add_scalar('validation/hh/epoch_true_accuracy_hh', accuracy_hh, epoch)
        summary_writer.add_scalar('validation/hh/epoch_true_auc_hh', avg_auc_score_hh, epoch)

    return epoch_avg_loss, accuracy_p, epoch_duration


def plot_roc(known_scores, unknown_scores, cum_study_number=None):
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

    if cum_study_number == None:
        return avg_auc_score,accuracy, sens, spec, optimal_threshold
    else:
        # compute patient level stat
        p_dict_pred = {}
        # pred
        for i in range(y_true.shape[0]):
            if cum_study_number[i] in p_dict_pred.keys() and p_dict_pred[cum_study_number[i]] != 1.0:
                p_dict_pred[cum_study_number[i]] = y_pred[i]
            elif cum_study_number[i] not in p_dict_pred.keys():
                p_dict_pred[cum_study_number[i]] = y_pred[i]
        # True
        p_dict_ture = {}
        for i in range(y_true.shape[0]):
            if cum_study_number[i] in p_dict_ture.keys() and p_dict_ture[cum_study_number[i]] != 1.0:
                p_dict_ture[cum_study_number[i]] = y_true[i]
            elif cum_study_number[i] not in p_dict_ture.keys():
                p_dict_ture[cum_study_number[i]] = y_true[i]
        p_correct = 0
        y_true_p = np.zeros((len(p_dict_ture.keys())))
        y_pred_p = np.zeros((len(p_dict_ture.keys())))

        for j, key in enumerate(p_dict_ture.keys()):
            y_true_p[j] = p_dict_ture[key]
            y_pred_p[j] = p_dict_pred[key]
            if p_dict_pred[key] == p_dict_ture[key]:
                p_correct += 1
        p_accuracy = p_correct/len(p_dict_ture.keys())

        tn, fp, fn, tp = confusion_matrix(y_true_p, y_pred_p).ravel()
        sens_p = tp / (tp + fn)
        spec_p = tn / (tn + fp)

        return  avg_auc_score,accuracy, sens, spec, optimal_threshold, p_accuracy, sens_p, spec_p