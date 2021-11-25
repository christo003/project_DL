# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 10:31:48 2021

@author: Leo
"""

from dlc_practical_prologue import *
from matplotlib import pyplot as plt

import torchvision
from comparisonNets import comparisonNet1, comparisonNet2
print(len(generate_pair_sets(1)))

import sklearn
import random

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

compnet1 = comparisonNet1()
criterion1 = torch.nn.BCELoss()
optimizer1 = torch.optim.Adam(compnet1.parameters(), lr=0.001)

output = compnet1(train_image[0:1])

print('\nmodel parameters : \n')
for k in compnet1.parameters():
    print(k.size())


#%% Utils

def evaluateAccuracy(dataloader,model):
    incorrect_count = 0
    for img_input,target in iter(dataloader):        
        output = model(img_input)
        incorrect_count += sum(abs(target.view(-1,1)-output.round()))
        

    return (1-incorrect_count/len(dataloader.dataset))*100

#%% Training the model
print('\n ### Training Start ### \n')


dataset = list(zip(train_image,train_target))
random.shuffle(dataset)
dataset_train = dataset[:-300]
dataset_val = dataset[-300:]
dataset_test = list(zip(test_image,test_target))

batch_size = 100

train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)

start_training = time.time()
epochs = 50
for e in range(epochs):
    # for idx,img_input in enumerate(train_image):
    print('current epoch : %i/%i' % (e,epochs))
    for img_input,target in iter(train_loader):
        
        output = compnet1(img_input)
        loss1 = criterion1(output,target)
        
        compnet1.zero_grad()#reset the gradient for each mini-batch
        loss1.backward()
        optimizer1.step()
    
    
    
    print('Validation : %f %%' % (evaluateAccuracy(val_loader,compnet1)))
    #Il manque l'étape de validation a la fin de l'epochs
    
stop_training = time.time()

print('\nTraining took : %f seconds \n' % (stop_training-start_training))
#%% Test of the model

print('\n ### Testing Start ### \n')

print('Final score : %f %%' % (evaluateAccuracy(test_loader, compnet1)))


#%% Training of 2nd architecture

print('\n ### Training Start ### \n')

compnet2 = comparisonNet2()
criterion2 = torch.nn.BCELoss()
optimizer2 = torch.optim.Adam(compnet2.parameters(), lr=0.001)

start_training = time.time()
epochs = 50
for e in range(epochs):
    # for idx,img_input in enumerate(train_image):
    print('current epoch : %i/%i' % (e,epochs))
    for img_input,target in iter(train_loader):
        
        output = compnet2(img_input)
        loss2 = criterion2(output,target)
        
        compnet2.zero_grad()#reset the gradient for each mini-batch
        loss2.backward()
        optimizer2.step()
    
    
    
    print('Validation : %f %%' % (evaluateAccuracy(val_loader,compnet2)))
    #Il manque l'étape de validation a la fin de l'epochs
    
stop_training = time.time()

print('\nTraining took : %f seconds \n' % (stop_training-start_training))

#%% Test of the model
print('\n ### Testing Start ### \n')
print('Final score : %f %%' % (evaluateAccuracy(test_loader, compnet2)))

#%% 3rd architecture



