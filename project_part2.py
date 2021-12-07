#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Written by Christopher Straub
# inspired from Francois Fleuret <francois@fleuret.org> code (practical3)

import math
import torch
import numpy as np
import dlc_practical_prologue as prologue

torch.set_grad_enabled(False)


# In[2]:


class Module(object):
    def forward(self, *input):
        raise NotImplementedError
    def backward(self, *gradwrtoutput): 
        raise NotImplementedError
    def param(self): 
        return []


# In[3]:


class Linear(Module):
    def __init__(self,input_size, hidden_size):
        super().__init__()
        epsilon = 1e-6
          
        self.weights=(torch.empty( hidden_size,input_size).normal_(0,epsilon))#Weight
        self.biais=( torch.empty(hidden_size).normal_(0,epsilon)) #bias

    def sigma(self,x):
        return self.weights.mv(x)+self.biais
    def dsigma(self,x):
        return self.weights.t().mv(x)
    def param(self):
        return [self.weights,self.biais]
    def set_param(self,new_w,new_b):
        self.weights= new_w
        self.biais = new_b
        


# In[4]:


class Tanh(Module):
    def __init__(self):
        super().__init__()
    def sigma(self,x):
        return x.tanh()
    def dsigma(self,x):
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)


# In[5]:


class LossMSE(Module):  
    def __init__(self):
        super().__init__()
    def sigma(self,x,y):
        return (x - y).pow(2).sum() 
    def dsigma(self,x,y):
        return 2*(x - y)
   
        


# In[6]:


class Net(Module):
    def __init__(self):
        super().__init__()
        self.Parameters = []
        self.Activation = []
        self.dl_dw = []
        self.dl_db = []
    def forward(self,*input):
        train_input=input[0]
        x = train_input
        out_s = []
        out_x = [train_input]
        for i in range(len(self.Activation)):
            s = self.Parameters[i].sigma(x)
            out_s.append(s)
            x = self.Activation[i].sigma(s)
            out_x.append(x)
        return [out_x,out_s]

    def backward(self,*gradwrtoutput):
        train_target,layer_output,Loss = gradwrtoutput
        dl_dw = self.dl_dw
        dl_db = self.dl_db
        N=len(dl_dw)
        x,s=layer_output
        x0 = x[0]
        dl_dx2 = Loss.dsigma(x[N], train_target)
        dl_ds2 = self.Activation[N-1].dsigma(s[N-1]) * dl_dx2
        dl_dw2 = dl_dw[N-1]
        dl_db2 = dl_db[N-1]
        dl_dw2.add_(dl_ds2.view(-1, 1).mm(x[N-1].view(1, -1)))
        dl_db2.add_(dl_ds2)
        out_dl_dw = [dl_dw2]
        out_dl_db = [dl_db2]

        for i in range(1,N,1):

            dl_dx1 = self.Parameters[N-i].dsigma(dl_ds2)
            dl_ds1 = self.Activation[N-1-i].dsigma(s[N-1-i]) * dl_dx1
            dl_dw1 = dl_dw[N-1-i]
            dl_db1 = dl_db[N-1-i]
            dl_dw1.add_(dl_ds1.view(-1, 1).mm(x[N-1-i].view(1, -1)))
            dl_db1.add_(dl_ds1)
            out_dl_dw.insert(0,dl_dw1)
            out_dl_db.insert(0,dl_db1)
            dl_ds2 = dl_ds1
        self.dl_dw,self.dl_db = out_dl_dw,out_dl_db
    def param(self):
        return self.Parameters
    def init(self,new_Parameters,new_Activation):
        self.Parameters= new_Parameters
        self.Activation = new_Activation
        self.dl_dw = [torch.empty(p.param()[0].size()) for p in new_Parameters]
        self.dl_db = [torch.empty(p.param()[1].size()) for p in new_Parameters]
    def set_param(self,num_layer,new_w,new_b):
        self.Parameters[num_layer].set_param(new_w,new_b)
    def get_grad(self,num_layer):
        return [self.dl_dw[num_layer],self.dl_db[num_layer]]
    def zero_grad(self):
        for dw in self.dl_dw:
            dw.zero_()
        for db in self.dl_db:
            db.zero_()




# In[7]:


class Sequential(Module):
    def __init__(self,*layers):
        super().__init__()
        self.layers = layers 
    def init_net(self):
        net = Net()
        new_Parameters=[]
        new_Activation=[]
        i = 0
        for layer in self.layers : 
            if np.mod(i,2)==0:
                new_Parameters.append(layer)
            else : 
                new_Activation.append(layer)
            i=i+1
        net.init(new_Parameters,new_Activation)
        return net 


# In[13]:


train_input, train_target, test_input, test_target = prologue.load_data(one_hot_labels = True,
                                                                        normalize = True)

nb_classes = train_target.size(1)
nb_train_samples = train_input.size(0)

zeta = 0.90

train_target = train_target * zeta
test_target = test_target * zeta

nb_hidden = 50
eta = 1e-1 / nb_train_samples
epsilon = 1e-6


net = Sequential(Linear (train_input.size(1),(nb_hidden)),Tanh(),Linear( nb_hidden,nb_classes),Tanh()).init_net()

loss = LossMSE()

for k in range(1000):

    # Back-prop

    acc_loss = 0
    nb_train_errors = 0
    
    net.zero_grad()


    for n in range(nb_train_samples):
        input =  train_input[n]
        x,s = net.forward(input)
        
        x2 = x[-1]
        pred = x2.max(0)[1].item()
        if train_target[n, pred] < 0.5:
            nb_train_errors = nb_train_errors + 1
        acc_loss = acc_loss + loss.sigma(x2, train_target[n])
        
        
        gradwrtoutput = [train_target[n],[x,s],loss]
        net.backward(*gradwrtoutput)

    # Gradient step
   
    for i in range(len(net.param())):
        dl_dw ,dl_db = net.get_grad(i)
        new_w=net.param()[i].param()[0]-eta * dl_dw 
        new_b=net.param()[i].param()[1]-eta* dl_db 
        net.set_param(i,new_w,new_b)

    
    # Test error

    nb_test_errors = 0
    
    for n in range(test_input.size(0)):
        input = test_input[n]
        x,s = net.forward(input)
        x2 = x[-1]
        pred = x2.max(0)[1].item()
        if test_target[n, pred] < 0.5:
            nb_test_errors = nb_test_errors + 1

    print('{:d} acc_train_loss {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'
          .format(k,
                  acc_loss,
                  (100 * nb_train_errors) / train_input.size(0),
                  (100 * nb_test_errors) / test_input.size(0)))


# In[14]:



train_input.size()


# In[ ]:




