# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn

import torch.nn.functional as F

class comparisonNet(nn.Module):
    def __init__(self):
        super(comparisonNet,self).__init__()
        self.conv1 = nn.Conv2d(2,10,2)
        self.conv2 = nn.Conv2d(10,20,4)
        self.fc1 = nn.Linear(500,16)
        self.fc2 = nn.Linear(16,1)
        
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        # print(x.shape)
        #x = nn.BatchNorm2d(x)
        x = F.relu(F.max_pool2d(self.conv2(x),kernel_size = 2))
        # print(x.shape)
        x = F.relu(self.fc1(x.flatten()))
        # print(x.shape)
        x = F.sigmoid(self.fc2(x))
        # print(x.shape)
        return x
        