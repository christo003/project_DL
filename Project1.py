# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 10:31:48 2021

@author: Leo
"""

from dlc_practical_prologue import *
from matplotlib import pyplot as plt

import torchvision
from comparisonNets import comparisonNet, comparisonNetWS, comparisonNetWSAuxLoss
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


nbr_iteration = 10


#%% Utils

def evaluateAccuracy(dataloader,model):
    incorrect_count = 0
    for img_input,target,test in iter(dataloader):  
        mean,std = img_input.mean(),img_input.std()
        img_input.sub_(mean).div_(std)
        output = model(img_input)
        if isinstance(output,tuple):#ugly trick because one time we have one output, and after we have three output...to optimize
            output = output[2]
        incorrect_count += sum(abs(target.view(-1,1)-output.round()))
        

    return (1-incorrect_count/len(dataloader.dataset))*100



dataset = list(zip(train_image,train_target,train_classes))
random.shuffle(dataset)
dataset_train = dataset[:-300]
dataset_val = dataset[-300:]
dataset_test = list(zip(test_image,test_target,test_classes))

batch_size = 100


train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)


#%% Training the model
print('\n ### Training Start with 1st architecture : ### \n')


compnet1 = comparisonNet()
optimizer1 = torch.optim.Adam(compnet1.parameters(), lr=0.001)
criterion1 = torch.nn.BCELoss()


cross_val_1 = []
for i in range(nbr_iteration):
    start_training = time.time()
    epochs = 25
    val_acc_1 = []
    compnet1.initParameter() #Reset the parameters before each training
    for e in range(epochs):
        # for idx,img_input in enumerate(train_image):
        print('current epoch : %i/%i' % (e+1,epochs))
        for img_input,target,target_digit in iter(train_loader):
            
            #Normalization of the input batch
            mean,std = img_input.mean(),img_input.std()
            img_input.sub_(mean).div_(std)
            
            #Forward pass
            output = compnet1(img_input)
            
            
            #Loss calculation and weight updates
            loss = criterion1(output,target)
            compnet1.zero_grad()#reset the gradient for each mini-batch
            loss.backward()
            optimizer1.step()
        
        
        val_acc = evaluateAccuracy(val_loader,compnet1)
        val_acc_1.append(val_acc)
        print('Validation accuracy : %f %%' % (val_acc))
        
    
    stop_training = time.time()
    cross_val_1.append(evaluateAccuracy(val_loader,compnet1))
    print('\nTraining took : %f seconds, iteration %i / %i \n' % (stop_training-start_training,i+1,nbr_iteration))
#%% Test of the model

print('\n ### Testing Start ### \n')

print('Final score 1st architecture : %f %%' % (evaluateAccuracy(test_loader, compnet1)))

time.sleep(2)

#%% Training of 2nd architecture

print('\n ### Training Start with weight sharing ### \n')

compnet2 = comparisonNetWS()
criterion2 = torch.nn.BCELoss()
#list(model1.parameters()) + list(model2.parameters()
# optimizer2 = torch.optim.Adam(list(compnet2.parameters())+list(compnet2.commonNet.parameters()), lr=0.001)
optimizer2 = torch.optim.Adam(compnet2.parameters(), lr = 0.001)

cross_val_2 = []
for i in range(nbr_iteration):

    start_training = time.time()
    epochs = 25
    val_acc_2 = []
    # compnet2.commonNet.initParameter()#reset the parameters uniformly of the common network
    compnet2.initParameter()#reset the parameters uniformly 
    for e in range(epochs):
        # for idx,img_input in enumerate(train_image):
        print('current epoch : %i/%i' % (e+1,epochs))
        for img_input,target,target_digit in iter(train_loader):
            
            #Normalization of the input batch
            mean,std = img_input.mean(),img_input.std()
            img_input.sub_(mean).div_(std)
            
            #Forware pass
            output = compnet2(img_input)
            
            #Loss calculation and weight updates
            loss = criterion2(output,target)        
            compnet2.zero_grad()#reset the gradient for each mini-batch
            loss.backward()
            optimizer2.step()
        
        
        val_acc = evaluateAccuracy(val_loader,compnet2)
        val_acc_2.append(val_acc)
        print('Validation accuracy : %f %%' % (val_acc))
        
    stop_training = time.time()
    
    cross_val_2.append(evaluateAccuracy(val_loader,compnet2))
    print('\nTraining took : %f seconds, iteration %i / %i \n' % (stop_training-start_training,i+1,nbr_iteration))

#%% Test of the model
print('\n ### Testing Start ### \n')
print('Final score with weigth sharing : %f %%' % (evaluateAccuracy(test_loader, compnet2)))

#%% 3rd architecture


print('\n ### Training Start with weight sharing and auxiliary Loss ### \n')

compnet3 = comparisonNetWSAuxLoss()
criterion_digit_1 = torch.nn.BCELoss()
criterion_digit_2 = torch.nn.BCELoss()
criterion3 = torch.nn.BCELoss()
optimizer3 = torch.optim.Adam(compnet3.parameters(), lr=0.001)


epochs = 40

cross_val_3 = []
for i in range(nbr_iteration):
    start_training = time.time()
    
    # compnet3.commonNet.initParameter()#reset the parameters uniformly of the common network
    compnet3.initParameter()#reset the parameters uniformly 
    val_acc_3 = []
    for e in range(epochs):
        # for idx,img_input in enumerate(train_image):
        print('current epoch : %i/%i' % (e+1,epochs))
        for img_input,target,target_digit in iter(train_loader):
            
            #Normalization of the input batch
            mean,std = img_input.mean(),img_input.std()
            img_input.sub_(mean).div_(std)
            
            #Forward pass
            output1,output2,output = compnet3(img_input)
            
            #Loss calculation and weight updates
            target_digit1_oh = F.one_hot(target_digit[:,0].to(torch.int64)).float()
            target_digit2_oh = F.one_hot(target_digit[:,1].to(torch.int64)).float()
            
            
            loss_digit_1 = criterion_digit_1(output1,target_digit1_oh)
            loss_digit_2 = criterion_digit_2(output2,target_digit2_oh)
            loss_class = criterion3(output,target)        
            loss_final = loss_digit_1+loss_digit_2+loss_class
            
            compnet3.zero_grad()#reset the gradient for each mini-batch
            loss_final.backward()
            optimizer3.step()
        
        
        val_acc = evaluateAccuracy(val_loader,compnet3)
        val_acc_3.append(val_acc)
        print('Validation accuracy : %f %%' % (val_acc))
        #Il manque l'étape de validation a la fin de l'epochs
        
    stop_training = time.time()
    cross_val_3.append(evaluateAccuracy(val_loader,compnet3))
    print('\nTraining took : %f seconds, iteration %i / %i \n' % (stop_training-start_training,i+1,nbr_iteration))

#%% Test of the model
print('\n ### Testing Start ### \n')
print('Final score with weigth sharing and axiliary loss : %f %%' % (evaluateAccuracy(test_loader, compnet3)))

#%% Results

fig,ax = plt.subplots()
ax.set_title('Cross validation')
x = range(1,11)
ax.set_ylim([0, 100])
ax.set_xlabel('iteration')
ax.set_ylabel('% accuracy')
ax.plot(x,cross_val_1,x,cross_val_2,x,cross_val_3)
ax.legend(('classic conv net','weight sharing','weight sharing + auxiliary loss'))


