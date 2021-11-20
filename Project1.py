# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 10:31:48 2021

@author: Leo
"""

from dlc_practical_prologue import *
from matplotlib import pyplot as plt

import torchvision
from comparisonNet1 import comparisonNet
print(len(generate_pair_sets(1)))


import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.nn.functional as F

import time

# class comparisonNet(nn.Module):
#     def __init__(self):
#         super(comparisonNet,self).__init__()
#         self.conv1 = nn.Conv2d(2,2,10)
#         self.conv2 = nn.Conv2d(2,2,5)
#         #self.fc1 = nn.Linear()
        
        
#     def forward(self,x):
#         x = F.relu(F.max_pool2d(self.conv1(x),kernel_size = 2))
#         # x = F.relu(F.max_pool2d(self.))
#         return x

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

dataset = generate_pair_sets(1000)

train_image = dataset[train_input_id].float()
train_target = dataset[train_target_id].float()
train_classes = dataset[train_classes_id].float()

test_image = dataset[test_input_id].float()
test_target = dataset[test_target_id].float()
test_classes = dataset[test_classes_id].float()

compnet = comparisonNet()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(compnet.parameters(), lr=0.001)

output = compnet(train_image[0:1])

print('\nmodel parameters : \n')
for k in compnet.parameters():
    print(k.size())

#%% Training the model

print('\n ### Training Start ### \n')
batch_size = 1

train_loader = DataLoader(dataset=list(zip(train_image,train_target)), batch_size=batch_size, shuffle=True)

start_training = time.time()
epochs = 20
for e in range(epochs):
    # for idx,img_input in enumerate(train_image):
    print('current epoch : %i/%i' % (e,epochs))
    for img_input,target in iter(train_loader):
        
        output = compnet(img_input)
        loss = criterion(output,target)
        
        compnet.zero_grad()#reset the gradient for each mini-batch
        loss.backward()
        optimizer.step()
        
    #Il manque l'étape de validation a la fin de l'epochs
    
stop_training = time.time()

print('\nTraining took : %f seconds \n' % (stop_training-start_training))
#%% Test of the model

print('\n ### Testing Start ### \n')

test_loader = DataLoader(dataset=list(zip(test_image,test_target)), batch_size=batch_size, shuffle=True)

incorrect_count = 0
for img_input,target in iter(test_loader):        
    output = compnet(img_input)
    if (target-output.round()).item() != 0:
        incorrect_count += 1
    
print('Final score : %f %%' % ((1-incorrect_count/len(test_loader))*100))








