import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class LSTMTagger(nn.Module):

    def __init__(self, hidden_dim, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim


        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(hidden_dim, 2)

        #self.residual = nn.Sequential(conv1x1(15,512), nn.BatchNorm3d(512), nn.ReLU(inplace=True)) # BasicBlock(15, 512, downsample = nn.Sequential(conv1x1(15,512)))

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(15*2, 2)
        #self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self, sentence):
        #embeds = self.residual(sentence)
        #embeds = self.avgpool(embeds)
        embeds = sentence
        lstm_out, _ = self.lstm(embeds.transpose(0,1))
        vector = torch.flatten(lstm_out.transpose(0,1), 1)
        x = self.hidden2tag(vector)
        #tag_space = self.hidden2tag1(x)
        #tag_scores = F.log_softmax(tag_space, dim=1)
        return x

