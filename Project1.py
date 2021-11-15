# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 10:31:48 2021

@author: Leo
"""

from dlc_practical_prologue import *
from matplotlib import pyplot as plt

import torchvision
#from comparisonNet1 import comparisionNet
print(len(generate_pair_sets(1)))


import torch
import torchvision
import torch.nn as nn

import torch.nn.functional as F

class comparisonNet(nn.Module):
    def __init__(self):
        super(comparisonNet,self).__init__()
        self.conv1 = nn.Conv2d(2,2,10)
        self.conv2 = nn.Conv2d(2,2,5)
        #self.fc1 = nn.Linear()
        
        
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),kernel_size = 2))
        return x

def showDigit(digit):
    """
    This function allows to show a digit

    Parameters
    ----------
    digit : Tensor containing digit

    Returns
    -------
    None.

    """
    
    plt.imshow(digit.numpy())
    
    
train_input_id = 0 
train_target_id = 1
train_classes_id = 2
test_input_id = 3
test_target_id = 4
test_classes_id = 5

data = generate_pair_sets(1)

fig1,ax1 = plt.subplots()
ax1.imshow(data[train_input_id][0][0],cmap='gray')
ax1.set_title(data[train_classes_id][0][0].item())

fig2,ax2 = plt.subplots()
ax2.imshow(data[train_input_id][0][1],cmap='gray')
ax2.set_title(data[train_classes_id][0][1].item())
plt.show()


#%% Building training set

train = generate_pair_sets(100)

train_image = train[train_input_id]
train_target = train[train_target_id]
train_classes = train[train_classes_id]

compnet = comparisonNet()

output = compnet.forward(train_image[0:5])

print(output.shape)







