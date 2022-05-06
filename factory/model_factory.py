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
# Date Created: 2018-08-15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn

from models import resnet, wide_resnet, resnext, densenet,vgg,resnet_h,resnet_h_r
from models.i3d import InceptionI3D



def get_model(config):

    assert config.model in ['i3d', 'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet','vgg', 'resnet_h', 'resnet_h_r']
    print('Initializing {} model (num_classes={})...'.format(config.model, config.num_classes))

    if config.model == 'i3d':

        from models.i3d import get_fine_tuning_parameters

        model = InceptionI3D(
            num_classes=config.num_classes,
            spatial_squeeze=True,
            final_endpoint='logits',
            in_channels=3,
            dropout_keep_prob=config.dropout_keep_prob
        )

    elif config.model == 'resnet':

        assert config.model_depth in [18, 34, 50, 101, 152]

        if config.model_depth == 18:

            model = resnet.resnet18(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

        elif config.model_depth == 34:

            model = resnet.resnet34(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

        elif config.model_depth == 50:

            model = resnet.resnet50(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

        elif config.model_depth == 101:

            model = resnet.resnet101(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

    elif config.model == 'resnet_h':

        assert config.model_depth in [18, 34, 50, 101, 152]

        if config.model_depth == 18:

            model = resnet_h.resnet18(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

        elif config.model_depth == 34:

            model = resnet_h.resnet34(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

        elif config.model_depth == 50:

            model = resnet_h.resnet50(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

        elif config.model_depth == 101:

            model = resnet_h.resnet101(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

    elif config.model == 'resnet_h_r':

        assert config.model_depth in [18, 34, 50, 101, 152]

        if config.model_depth == 18:

            model = resnet_h_r.resnet18(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

        elif config.model_depth == 34:

            model = resnet_h_r.resnet34(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

        elif config.model_depth == 50:

            model = resnet_h_r.resnet50(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

        elif config.model_depth == 101:

            model = resnet_h_r.resnet101(
                num_classes=config.num_classes,
                pretrained=config.pretrained)


    elif config.model == 'vgg':

        assert config.model_depth in [7, 11, 13, 16, 19]


        if config.model_depth == 11:

            model = vgg.vgg11_bn(
                num_classes=config.num_classes,
                pretrained=config.pretrained)
        
        elif config.model_depth == 7:

            model = vgg.vgg7_bn(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

        elif config.model_depth == 13:

            model = vgg.vgg13_bn(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

        elif config.model_depth == 16:

            model = vgg.vgg16_bn(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

        elif config.model_depth == 19:

            model = vgg.vgg19_bn(
                num_classes=config.num_classes,
                pretrained=config.pretrained)


    elif config.model == 'wideresnet':

        assert config.model_depth in [50]
        from models.wide_resnet import get_fine_tuning_parameters

        if config.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                k=config.wide_resnet_k,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

    elif config.model == 'resnext':

        assert config.model_depth in [50, 101, 152]
        from models.resnext import get_fine_tuning_parameters

        if config.model_depth == 50:
            model = resnext.resnet50(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                cardinality=config.resnext_cardinality,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 101:
            model = resnext.resnet101(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                cardinality=config.resnext_cardinality,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 152:
            model = resnext.resnet152(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                cardinality=config.resnext_cardinality,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

    elif config.model == 'densenet':

        assert config.model_depth in [121, 169, 201]

        if config.model_depth == 121:
            model = densenet.densenet121(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

        elif config.model_depth == 169:
            model = densenet.densenet169(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

        elif config.model_depth == 201:
            model = densenet.densenet201(
                num_classes=config.num_classes,
                pretrained=config.pretrained)

        elif config.model_depth == 161:
            model = densenet.densenet161(
                num_classes=config.num_classes,
                pretrained=config.pretrained)


    if 'cuda' in config.device:

        print('Moving model to CUDA device...')
        # Move model to the GPU
        model = model.cuda()

        if config.model != 'i3d':
            model = nn.DataParallel(model, device_ids=None)

        if config.checkpoint_path:

            print('Loading pretrained model {}'.format(config.checkpoint_path))
            assert os.path.isfile(config.checkpoint_path)

            checkpoint = torch.load(config.checkpoint_path)

            pretrained_weights = checkpoint['state_dict']
            
            model.load_state_dict(pretrained_weights,strict=False)
            
            return model, model.parameters()
            
    else:
        print('CPU training not supported.')
        print('only for saliency map extraction')
        return model, ''

    return model, model.parameters()
