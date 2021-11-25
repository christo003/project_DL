# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn

import torch.nn.functional as F

class comparisonNet1(nn.Module):
    def __init__(self):
        super(comparisonNet1,self).__init__()
        self.conv1 = nn.Conv2d(2,10,2)
        self.conv2 = nn.Conv2d(10,20,4)
        self.fc1 = nn.Linear(500,20)
        self.fc2 = nn.Linear(20,1)
        
        
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
        

class comparisonNet2(nn.Module):
    def __init__(self):
        super(comparisonNet2,self).__init__()
        self.commonNet = nn.Sequential(
            nn.Conv2d(1,5,2),
            nn.ReLU(),
            nn.Conv2d(5,10,4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.fc1 = nn.Linear(500,20)
        self.fc2 = nn.Linear(20,1)
                
    def forward(self,x):
        number1 = x[:,0,:,:].view(1,1,14,14)
        number2 = x[:,1,:,:].view(1,1,14,14)
        out1 = self.commonNet(number1)
        out2 = self.commonNet(number2)
        # print(out1.shape)
        # print(out2.shape)
        # print(out1.flatten().shape)
        # print(out2.flatten().shape)
        out = torch.cat((out1.flatten(),out2.flatten()))
        # print(out.shape)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out
    
    
# class numberRecognizerNet(nn.Module):
#     def __init__(self):
#         super(numberRecognizerNet)
        