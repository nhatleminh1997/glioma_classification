import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

conv = nn.Conv2d(3, 3, 3)  # Input dim is 3, output dim is 3
weight1= conv.weight
deconv = nn.ConvTranspose2d(3, 3, 3)
weight2= deconv.weight
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)

