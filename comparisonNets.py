# -*- coding: utf-8 -*-

import torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F

class comparisonNet(nn.Module):
    """
    This modele simply use convolutional layer and extract features in order
    to predict if one number is bigger than the other.
    
    It consider the input as a 2 channel image.
    """
    def __init__(self):
        super(comparisonNet,self).__init__()
        self.conv1_channel = 32
        self.conv2_channel = 64
        self.conv1 = nn.Conv2d(2,self.conv1_channel,2)
        self.conv1_bn=nn.BatchNorm2d(self.conv1_channel)
        self.conv2 = nn.Conv2d(self.conv1_channel,self.conv2_channel,4)
        self.conv2_bn=nn.BatchNorm2d(self.conv2_channel)
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

    def initParameter(self):
        self.conv1.weight.data = (torch.rand(self.conv1.weight.size())-0.5)*(1/math.sqrt(2))
        self.conv2.weight.data = (torch.rand(self.conv2.weight.size())-0.5)*(1/math.sqrt(self.conv1_channel))
        self.fc1.weight.data = (torch.rand(self.fc1.weight.size())-0.5)*(1/math.sqrt(1600))
        self.fc2.weight.data = (torch.rand(self.fc2.weight.size())-0.5)*(1/math.sqrt(20))
        

class  commonNet(nn.Module):
    def __init__(self):
        super(commonNet,self).__init__()
        self.conv1_channel = 16
        self.conv2_channel = 32
        self.conv1 = nn.Conv2d(1,self.conv1_channel,2)#2
        self.conv1_bn = nn.BatchNorm2d(self.conv1_channel)
        self.conv2 = nn.Conv2d(self.conv1_channel,self.conv2_channel,4)
        self.conv2_bn = nn.BatchNorm2d(self.conv2_channel)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(800,10)
        
        
    def forward(self,x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)),kernel_size = 2))
        x = F.softmax(self.fc1(self.flat(x)))
        return x
    
    def initParameter(self):
        self.conv1.weight.data = (torch.rand(self.conv1.weight.size())-0.5)*(1/math.sqrt(1))
        self.conv2.weight.data = (torch.rand(self.conv2.weight.size())-0.5)*(1/math.sqrt(self.conv1_channel))
        self.fc1.weight.data = (torch.rand(self.fc1.weight.size())-0.5)*(1/math.sqrt(800))
            



class comparisonNetWS(nn.Module):
    """
    This modele use a sub network that allow to classifiy each number before make the 
    decision if one number is bigger than the other.
    
    The input of each network is a number with 1 channel.
    """
    def __init__(self):
        super(comparisonNetWS,self).__init__()
        self.conv1_channel = 16
        self.conv2_channel = 32
        self.commonNet = nn.Sequential(
            nn.Conv2d(1,self.conv1_channel,2),#2
            nn.BatchNorm2d(self.conv1_channel),
            nn.ReLU(),
            nn.Conv2d(self.conv1_channel,self.conv2_channel,4),#4
            nn.BatchNorm2d(self.conv2_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(800,10),
            nn.Softmax()
            )
        # self.commonNet = commonNet()
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
        
        return out
    
    def initParameter(self):
        self.fc1.weight.data = (torch.rand(self.fc1.weight.size())-0.5)*(1/math.sqrt(20))
        self.fc2.weight.data = (torch.rand(self.fc2.weight.size())-0.5)*(1/math.sqrt(30))

class comparisonNetWSAuxLoss(nn.Module):
    """
    This modele use a sub network that allow to classifiy each number before make the 
    decision if one number is bigger than the other.
    
    In addition to that the forward method output the intermediate result
    in order to build a loss that take into account the capacity of the network
    to find the number in each channel.
    """
    def __init__(self):
        super(comparisonNetWSAuxLoss,self).__init__()
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
        # self.commonNet = commonNet()
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
    
    def initParameter(self):
        self.fc1.weight.data = (torch.rand(self.fc1.weight.size())-0.5)*(1/math.sqrt(20))
        self.fc2.weight.data = (torch.rand(self.fc2.weight.size())-0.5)*(1/math.sqrt(30))        
    
    

        