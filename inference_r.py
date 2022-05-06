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
from torch.nn import ReLU

from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)
from utils.utils import *
import factory.data_factory as data_factory
import factory.model_factory as model_factory
from config_infer import parse_opts
import numpy as np
import pandas as pd
from torch.nn import functional as F
import cv2
from skimage.transform import rotate
from PIL import Image, ImageDraw
import SimpleITK as sitk

def main():
    ####################################################################
    ####################################################################
    # Configuration and logging
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = parse_opts()
    # set inference parameters
    config.video_path = '/mass/pengfei/recurrece_prediction/post_data_2020_0219_2d_all_slice_all_label'
    config.save_dir = '/mass/pengfei/recurrece_prediction/output/inference/analysis'
    config.dataset = 'MRI_H_R'
    config.apply_mask = 'False'
    config.apply_ls = True
    config.Hierarchical = True
    config.in_chanel = 5
    config.model = 'resnet_h'
    config.model_depth = 18
    config.pretrained = False
    config.batch_size = 1
    config.checkpoint_path = '/mass/pengfei/recurrece_prediction/output/hierarchical_beta_ls_beta/w_apt/20200426_1813_MRI_H_resnet_h_18_lr0.00010/checkpoints/0.8357142857142857_69_save_best.pth'

    #'/mass/pengfei/recurrece_prediction/output/hierarchical_beta_ls/wo_apt/20200408_1536_MRI_H_resnet_h_18_lr0.00010/checkpoints/0.8119047619047619_71_save_best.pth'
    config.manual_seed = 666

    config = prepare_output_dirs(config)

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

    ####################################################################
    ####################################################################

    train_transforms = {
        'spatial': None,
        'temporal': None,
        'target':None
    }

    validation_transforms = {
        'spatial': None,
        'temporal': None,
        'target': None
    }

    ####################################################################
    ####################################################################
    # Setup of data pipeline

    data_loaders = data_factory.get_data_loaders(config, train_transforms, validation_transforms)
    phases = ['train', 'validation'] if 'validation' in data_loaders else ['train']
    print('#' * 60)

    ####################################################################
    ####################################################################

    model.eval()

    # Epoch statistics
    steps_in_epoch = int(np.ceil(len(data_loaders['validation'].dataset) /config.batch_size))
    losses = 0
    accuracies = np.zeros(steps_in_epoch, np.float32)

    epoch_start_time = time.time()
    count_total = 0
    count_correct = 0

    for step, (clips, targets, loss_mask, targets_h, targets_hh, study_number, targets_p) in enumerate(data_loaders['validation']):
        clips_ = torch.squeeze(clips, dim=0)
        targets_ = torch.squeeze(targets, dim=0)
        loss_mask_ = torch.squeeze(loss_mask, dim=0)
        targets_h_ = torch.squeeze(targets_h, dim=0)
        targets_hh_ = torch.squeeze(targets_hh, dim=0)
        targets_p_ = torch.squeeze(targets_p, dim=0)

        for num in range(15):
            study_number_running = study_number[0]
            running_save_dir = os.path.join(config.save_dir,study_number_running + '_' + str(targets_p_.numpy()[0]))
            if not os.path.exists(running_save_dir):
                os.mkdir(running_save_dir)

            file_name_to_export = study_number_running + '_' + str(num)

            # Move inputs to GPU memory
            clips = clips_[num,:,:,:].cuda().unsqueeze(0)
            targets = targets_[num,:].cuda().view(-1)
            loss_mask = loss_mask_[num,:].cuda().view(-1)
            targets_h = targets_h_[num,:].cuda().view(-1)
            targets_hh = targets_hh_[num,:].cuda().view(-1)


            finalconv_name = 'layer4'
            classes = {0: 'negative',
                       1: 'postive'}
            # hook the feature extractor
            features_blobs = []
            def hook_feature(module, input, output):
                features_blobs.append(output.data.cpu().numpy())
            model.module._modules.get(finalconv_name).register_forward_hook(hook_feature)
            # get the softmax weight
            params = list(model.parameters())
            weight_softmax = np.squeeze(params[-6].cpu().data.numpy())  # fc for recurrent or not

            # Feed-forward through the network
            logits, logits_h, logits_hh, x_subs = model.forward(clips)

            # output of learnable subtraction layer
            x_subs = x_subs.squeeze().detach().cpu().numpy()  # 0 T1c - T1  1 T2- FLAIR
            origin = clips.squeeze().detach().cpu().numpy()

            t1c_t1_learn = (denormalize(x_subs[0]) * 255.0).astype(np.uint8)
            t2_flair_learn = (denormalize(x_subs[1]) * 255.0).astype(np.uint8)

            t1c_t1_learn = np.rot90(t1c_t1_learn, 3)
            t1c_t1_learn = np.fliplr(t1c_t1_learn)
            t2_flair_learn = np.rot90(t2_flair_learn, 3)
            t2_flair_learn = np.fliplr(t2_flair_learn)
            if config.in_chanel == 5:
                t1c_t1 = (denormalize(origin[4] - origin[1]) * 255.0).astype(np.uint8)
                t2_flair = (denormalize(origin[3] - origin[2]) * 255.0).astype(np.uint8)

                t1c_t1 = np.rot90(t1c_t1, 3)
                t1c_t1 = np.fliplr(t1c_t1)
                t2_flair = np.rot90(t2_flair, 3)
                t2_flair = np.fliplr(t2_flair)
            else:
                t1c_t1 = (denormalize(origin[3] - origin[0]) * 255.0).astype(np.uint8)
                t2_flair = (denormalize(origin[2] - origin[1]) * 255.0).astype(np.uint8)

                t1c_t1 = np.rot90(t1c_t1, 3)
                t1c_t1 = np.fliplr(t1c_t1)
                t2_flair = np.rot90(t2_flair, 3)
                t2_flair = np.fliplr(t2_flair)

            _, preds = torch.max(logits, 1)

            #### CAM calculation start
            h_x = F.softmax(logits, dim=1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            probs = probs.cpu().numpy()
            idx = idx.cpu().numpy()
            # output the prediction
            true = classes[targets.cpu().numpy()[0]]
            for i in range(2):
                print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
            print('True: ', true)
            #if true == classes[idx[0]]: #correct pred
            # generate class activation mapping for the top1 prediction
            CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
            # else:
            #     # generate class activation mapping for the wrong prediction
            #     CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
            height, width, _ = 256, 256, 3
            # render the CAM and output
            print('output ' + file_name_to_export + '_CAM.jpg' + ' for the top1 prediction: %s' % classes[idx[0]])

            img_array = np.squeeze(clips.cpu().numpy())
            if img_array.shape[0] == 5:
                T1c = ((img_array[-1, :, :] + 1) / 2 * 255).astype(np.uint8)
            elif img_array.shape[0] == 4:
                T1c = ((img_array[-1, :, :] + 1) / 2 * 255).astype(np.uint8)
            else:
                print('Error!')
                break
            # save original cam
            cam_or = cv2.resize(CAMs[0], (width, height))
            # give up out brain region
            cam_or_mask = fill_small_holes((T1c > 0).astype(np.uint8))
            cam_or_masked = np.multiply(cam_or, cam_or_mask)
            index = height * width - np.sum(cam_or_mask)
            cam_or = np.rot90(cam_or_masked, 3)
            cam_or = np.fliplr(cam_or)
            itk_cam = sitk.GetImageFromArray(cam_or)
            save_path = os.path.join(running_save_dir, file_name_to_export + '_CAM_or_masked.nii')
            sitk.WriteImage(itk_cam, save_path)

            img = cv2.cvtColor(T1c, cv2.COLOR_GRAY2RGB)

            # find 95 percentle mask
            array = CAMs[0].flatten()
            sorted_array = np.sort(array)
            mask = (CAMs[0] >= sorted_array[int(sorted_array.shape[0] * 0.90)]).astype(np.uint8)
            # resize mask to out shape
            mask = cv2.resize(mask, (width, height))
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
            # apply mask to heatmap
            heatmap_masked = np.zeros((height, width, 3))
            for i in range(heatmap.shape[2]):
                heatmap_masked[:, :, i] = np.multiply(heatmap[:, :, i], mask)

            result = heatmap_masked * 0.5 + img * 0.5
            # match the direction
            data = rotate(result, 270, resize=False)
            result = np.fliplr(data).astype(np.uint8)
            save_path = os.path.join(running_save_dir, file_name_to_export + '_CAM_masked.jpg')

            cv2_im_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im_rgb)
            draw = ImageDraw.Draw(pil_im)

            # Draw the text
            #draw.text((0, 0), '{:.5f} -> {}'.format(probs[0], classes[idx[0]]))
            # Save the image
            pil_im.save(save_path)

            # save original heatmap
            result = heatmap * 0.5 + img * 0.5
            # match the direction
            data = rotate(result, 270, resize=False)
            result = np.fliplr(data).astype(np.uint8)
            save_path = os.path.join(running_save_dir, file_name_to_export + '_CAM.jpg')

            cv2_im_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im_rgb)
            draw = ImageDraw.Draw(pil_im)
            # Draw the text
            #draw.text((0, 0), '{:.5f} -> {}'.format(probs[0], classes[idx[0]]))
            # Save the image
            pil_im.save(save_path)

            # output of ls layer
            pil_im = Image.fromarray(t1c_t1)
            save_path = os.path.join(running_save_dir, file_name_to_export + '_t1c_t1.jpg')
            pil_im.save(save_path)

            pil_im = Image.fromarray(t1c_t1_learn)
            save_path = os.path.join(running_save_dir, file_name_to_export + '_t1c_t1_learn.jpg')
            pil_im.save(save_path)

            pil_im = Image.fromarray(t2_flair)
            save_path = os.path.join(running_save_dir, file_name_to_export + '_t2_flair.jpg')
            pil_im.save(save_path)

            pil_im = Image.fromarray(t2_flair_learn)
            save_path = os.path.join(running_save_dir, file_name_to_export + '_t2_flair_learn.jpg')
            pil_im.save(save_path)

            # Guided backprop
            GBP = GuidedBackprop(model)
            # Get gradients
            #temp = np.zeros((1,256,256))
            guided_grads = GBP.generate_gradients(clips, int(targets.cpu().numpy()[0]))
            #temp[0] = guided_grads[0]
            grayscale_guided_grads = convert_to_grayscale(guided_grads)
            grayscale_guided_grads = np.rot90(np.squeeze(grayscale_guided_grads), 3)
            grayscale_guided_grads = (np.fliplr(grayscale_guided_grads)*255).astype(np.uint8)
            img = cv2.applyColorMap(cv2.resize(grayscale_guided_grads, (width, height)), cv2.COLORMAP_JET)
            # Save colored gradients
            cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im_rgb)
            save_path = os.path.join(running_save_dir, file_name_to_export + '_GB.jpg')
            pil_im.save(save_path)


            #### CAM calculation end

            # Calculate accuracy
            correct = torch.sum(preds == targets.data)
            accuracy = correct.double() / clips.size()[0]
            count_total += clips.size()[0]
            count_correct += correct.double()

            # Save statistics
            accuracies[step] = accuracy.item()
            # extract CAM


            if step == 0 and num == 0:
                cum_study_number = [study_number_running]

                cum_targets = targets.detach().cpu()
                cum_scores = torch.nn.functional.softmax(logits.detach().cpu(), dim=1)

                cum_targets_h = targets_h.detach().cpu()
                cum_scores_h = torch.nn.functional.softmax(logits_h.detach().cpu(), dim=1)

                cum_targets_hh = targets_hh.detach().cpu()
                cum_scores_hh = torch.nn.functional.softmax(logits_hh.detach().cpu(), dim=1)
            else:
                cum_study_number += [study_number_running]

                cum_targets = torch.cat((cum_targets, targets.detach().cpu()), 0)
                cum_scores = torch.cat((cum_scores, torch.nn.functional.softmax(logits.detach().cpu(), dim=1)), 0)

                cum_targets_h = torch.cat((cum_targets_h, targets_h.detach().cpu()), 0)
                cum_scores_h = torch.cat((cum_scores_h, torch.nn.functional.softmax(logits_h.detach().cpu(), dim=1)), 0)

                cum_targets_hh = torch.cat((cum_targets_hh, targets_hh.detach().cpu()), 0)
                cum_scores_hh = torch.cat((cum_scores_hh, torch.nn.functional.softmax(logits_hh.detach().cpu(), dim=1)), 0)



    # Epoch statistics

    avg_auc_score, accuracy, sens, spec, optimal_threshold, p_accuracy, sens_p, spec_p = plot_roc(cum_targets,
                                                                                                  cum_scores,
                                                                                                  cum_study_number)

    avg_auc_score_h, accuracy_h, sens_h, spec_h, optimal_threshold_h = plot_roc(cum_targets_h, cum_scores_h)

    avg_auc_score_hh, accuracy_hh, sens_hh, spec_hh, optimal_threshold_hh = plot_roc(cum_targets_hh, cum_scores_hh)

    print('auc: ',avg_auc_score,'acc: ',accuracy, 'sens: ', sens, 'spec: ', spec, 'threshold: ', optimal_threshold)

def denormalize(x):
    max = np.max(x)
    min = abs(np.min(x))
    new_x = ( x + min) / (max + min)
    return new_x

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
        return avg_auc_score,accuracy, sens, spec, optimal_threshold, p_accuracy, sens_p, spec_p

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def fill_small_holes (im_th):
    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out.astype(np.bool)

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = self.model.module.conv1

        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.module.layer1._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)
        for pos, module in self.model.module.layer2._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)
        for pos, module in self.model.module.layer3._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)
        for pos, module in self.model.module.layer4._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop

        one_hot_output = torch.FloatTensor(1, model_output[0].size()[-1]).zero_().cuda()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output[0].backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.cpu().data.numpy()[0]
        return gradients_as_arr


if __name__ == '__main__':
    main()