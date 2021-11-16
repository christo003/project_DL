#!/usr/bin/env python
# coding: utf-8

# In[1]:



from torch import nn
from torch.nn import functional as F
import torch
import math

from torch import optim
from torch import Tensor
from torch import nn

from dlc_practical_prologue import generate_pair_sets


# ### importation données

# In[2]:


N=100
train_input,train_target,train_classes,test_input,test_target,test_classes=generate_pair_sets(N)


# ### affichage des données pour compréhension

# In[3]:


import matplotlib.pyplot as plt
if N >10:
    k=5
for k in range(k):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('im1<=im2 : '+str(bool(train_target[k].item())))
    ax1.imshow(train_input[k][0])
    ax2.imshow(train_input[k][1])
    plt.show()
    


# ### c'est quoi l'architecture d'un reseau de neurone ? c'est le nombre de hidden layers et le nombre poids ? 
#  est ce que le loss fait partie de l'architecture ?
#  
#  est ce que le type de resolution descente du gradient ou adam etc fait partie de l'architecture ?
#  
#  est ce que le batch size ou le nombre epoch fait partie de l'architecture ? 
#  

# ### comment choisir son architeture ? 

# ### comment construire un RN?

# In[4]:


hidden_units=2
eta, mini_batch_size = 1e-1, 2


# In[5]:


def train_model(model, train_input, train_target):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1)
    nb_epochs = 250

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()


# In[6]:


def compute_nb_errors(model, data_input, data_target):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors
            


# In[7]:


target=torch.nn.functional.one_hot(train_target,-1).view(N,-1)
print(target.view(N,-1).size())
print(train_input.view(N, -1).size())
b=3


# In[8]:


model=nn.Sequential(nn.Linear(2*14*14,500),nn.ReLU(),nn.Linear(500,2))
train_model(model, train_input.view(N,-1), train_target)


# In[9]:


compute_nb_errors(model, test_input.view(N,-1), test_target)


# comment les convolutions network fonctionnent? 
# 

# à faire implémenter avec un convolution network!
