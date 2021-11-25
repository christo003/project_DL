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





#%% Utils

def evaluateAccuracy(dataloader,model):
    incorrect_count = 0
    for img_input,target,test in iter(dataloader):        
        output = model(img_input)
        if isinstance(output,tuple):
            output = output[2]#trick because one time we have one output, and after we have three output...to optimize
        incorrect_count += sum(abs(target.view(-1,1)-output.round()))
        

    return (1-incorrect_count/len(dataloader.dataset))*100

#%% Training the model
print('\n ### Training Start ### \n')


dataset = list(zip(train_image,train_target,train_classes))
random.shuffle(dataset)
dataset_train = dataset[:-300]
dataset_val = dataset[-300:]
dataset_test = list(zip(test_image,test_target,test_classes))

batch_size = 100

compnet1 = comparisonNet1()
optimizer1 = torch.optim.Adam(compnet1.parameters(), lr=0.001)


criterion1 = torch.nn.BCELoss()


train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)

start_training = time.time()
epochs = 25
for e in range(epochs):
    # for idx,img_input in enumerate(train_image):
    print('current epoch : %i/%i' % (e,epochs))
    for img_input,target,target_digit in iter(train_loader):
        
        output = compnet1(img_input)
        
        loss = criterion1(output,target)
        compnet1.zero_grad()#reset the gradient for each mini-batch
        loss.backward()
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
criterion_digit_1 = torch.nn.BCELoss()
criterion_digit_2 = torch.nn.BCELoss()
criterion2 = torch.nn.BCELoss()
optimizer2 = torch.optim.Adam(compnet2.parameters(), lr=0.001)

start_training = time.time()
epochs = 50
for e in range(epochs):
    # for idx,img_input in enumerate(train_image):
    print('current epoch : %i/%i' % (e,epochs))
    for img_input,target,target_digit in iter(train_loader):
        
        output1,output2,output = compnet2(img_input)
        
        target_digit1_oh = F.one_hot(target_digit[:,0].to(torch.int64)).float()
        target_digit2_oh = F.one_hot(target_digit[:,1].to(torch.int64)).float()

        
        loss_digit_1 = criterion_digit_1(output1,target_digit1_oh)
        loss_digit_2 = criterion_digit_2(output2,target_digit2_oh)
        loss_class = criterion2(output,target)        
        loss_final = loss_digit_1+loss_digit_2+loss_class
        
        compnet2.zero_grad()#reset the gradient for each mini-batch
        loss_final.backward()
        optimizer2.step()
    
    
    
    print('Validation : %f %%' % (evaluateAccuracy(val_loader,compnet2)))
    #Il manque l'étape de validation a la fin de l'epochs
    
stop_training = time.time()

print('\nTraining took : %f seconds \n' % (stop_training-start_training))

#%% Test of the model
print('\n ### Testing Start ### \n')
print('Final score : %f %%' % (evaluateAccuracy(test_loader, compnet2)))

#%% 3rd architecture



