# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn

import torch.nn.functional as F

class comparisionNet(nn.Module):
    def __init__(self):
        super(comparisionNet,self).__init__()
        self.conv1 = nn.Conv2d(2,2,10)
        self.conv2 = nn.Conv2d(2,2,5)
        self.fc1 = nn.Linear()
        
        
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),kernel_size = 2))
        return x
        