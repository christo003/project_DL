# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn

import torch.nn.functional as F

class comparisonNet1(nn.Module):
    def __init__(self):
        super(comparisonNet1,self).__init__()
        conv1_channel = 32
        conv2_channel = 64
        self.conv1 = nn.Conv2d(2,conv1_channel,2)
        self.conv1_bn=nn.BatchNorm2d(conv1_channel)
        self.conv2 = nn.Conv2d(conv1_channel,conv2_channel,4)
        self.conv2_bn=nn.BatchNorm2d(conv2_channel)
        self.flat = nn.Flatten()
        
        
        self.fc1 = nn.Linear(1600,20)
        self.fc2 = nn.Linear(20,1)
        
        
    def forward(self,x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        
        # print(x.shape)
        #x = nn.BatchNorm2d(x)
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)),kernel_size = 2))
        # print(x.shape)
        x = F.relu(self.fc1(self.flat(x)))
        # print(x.shape)
        x = F.sigmoid(self.fc2(x))
        # print(x.shape)
        return x
        

class comparisonNet2(nn.Module):
    def __init__(self):
        super(comparisonNet2,self).__init__()
        conv1_channel = 16
        conv2_channel = 32
        self.commonNet = nn.Sequential(
            nn.Conv2d(1,conv1_channel,2),#2
            nn.BatchNorm2d(conv1_channel),
            nn.ReLU(),
            nn.Conv2d(conv1_channel,conv2_channel,4),#4
            nn.BatchNorm2d(conv2_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(800,10),
            nn.Softmax()
            )
        self.fc1 = nn.Linear(20,30)
        self.fc2 = nn.Linear(30,1)
                
    def forward(self,x):
        number1 = x[:,0,:,:].view(-1,1,14,14)
        number2 = x[:,1,:,:].view(-1,1,14,14)
        out1 = self.commonNet(number1)
        #out1 len(out1) = 10
        out2 = self.commonNet(number2)
        
        # Final loss = gamma*MSE1 + gamma*MSE2 + alpha*FINAL DECISION ()
        # print(out1.shape)
        # print(out2.shape)
        # print(out1.flatten().shape)
        # print(out2.flatten().shape)
        out = torch.cat((out1,out2),1)
        # print(out.shape)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        
        return out1,out2,out
    
    
class numberRecognizerNet(nn.Module):
    def __init__(self):
        super(numberRecognizerNet,self).__init__()
        self.conv1 = nn.Conv2d(2,10,2)
        self.conv2 = nn.Conv2d(10,20,4)
        