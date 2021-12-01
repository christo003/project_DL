#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Written by Christopher Straub
# inspired from Francois Fleuret <francois@fleuret.org> code (practical3)

import math
import torch

import dlc_practical_prologue as prologue


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
        #self.p = []   
        #self.p.append(torch.empty(input_size, hidden_size).normal_())#Weight
        #self.p.append( torch.zeros(hidden_size)) #bias
        self.input_size = input_size
        self.hidden_size = hidden_size
    def sigma(self,weights,biais,x):
        return weights.mv(x)+biais
    def dsigma(self,weights,biais,x):
        return weights.t().mv(x)
    #def param(self):
    #    return [self,input_size, hidden_size]


# In[4]:


class Tanh(Module):
    def __init__(self,*input):
        super().__init__()
    def sigma(self,x):
        return x.tanh()
    def dsigma(self,x):
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)


# In[5]:


class LossMSE(Module): #le loss doit il appartenir au module ? 
    def __init__(self):#,*net):
        super().__init__()
        #self.net=net
    def sigma(self,x,y):
        return (x - y).pow(2).sum() 
    def dsigma(self,x,y):
        return 2*(x - y)
    #def backward(self,*gradwrtoutput):
    #    net=self.net[0]
    #    net.gradwrtoutput[-1].add_(self.dloss(net.forward(*net.train),*net.target))
        
        


# In[6]:



def forward(input):#(w1, b1, w2, b2, x):
    #print(len(input))
    Weights,Biais,Activation, train_input=input

    x = train_input
    out_s = []
    out_x = [train_input]
    for i in range(len(Activation)):
        s = Weights[i].mv(x)+Biais[i]
        out_s.append(s)
        x = Activation[i].sigma(s)
        out_x.append(x)
    #print(x.size())
    #print(Weights[-1].size())
    #print(Biais[-1].size())
    #s = Weights[-1].mv(x)+Biais[-1]
    return [out_x,out_s]

def backward(gradwrtoutput):#(w1, b1, w2, b2,t,x, s1, x1, s2, x2,dl_dw1, dl_db1, dl_dw2, dl_db2):
    #print(len(gradwrtoutput))
    Weights,Biais,Activation,train_target,layer_output,dl_dw,dl_db,Loss = gradwrtoutput.copy()
    N=len(dl_dw)-1
    print(N)
    #print(layer_output)
    x,s=layer_output
    x0 = x[0]
    #print(x[N+1])
    dl_dx2 = Loss.dsigma(x[N+1], train_target)
    dl_ds2 = Activation[N].dsigma(s[N]) * dl_dx2
    dl_dw2 = dl_dw[N]
    dl_db2 = dl_db[N]
    dl_dw2.add_(dl_ds2.view(-1, 1).mm(x[N].view(1, -1)))
    dl_db2.add_(dl_ds2)
    out_dl_dw = [dl_dw2]
    out_dl_db = [dl_db2]
    N=N-1 
    for i in range(N):
        #print(i)
        dl_dx1 = Weights[N-i].t().mv(dl_ds2)
        dl_ds1 = Activation[N-i].dsigma(s[N-i]) * dl_dx1
        dl_dw1 = dl_dw[N-i]
        dl_db1 = dl_db[N-i]
        dl_dw1.add_(dl_ds1.view(-1, 1).mm(x[N-i].view(1, -1)))
        dl_db1.add_(dl_ds1)
        out_dl_dw.insert(0,dl_dw1)
        out_dl_db.insert(0,dl_db1)
        dl_ds2 = dl_ds1
    return out_dl_dw,out_dl_db


# In[9]:


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


w1 = torch.empty(nb_hidden, train_input.size(1)).normal_(0, epsilon)
b1 = torch.empty(nb_hidden).normal_(0, epsilon)
w2 = torch.empty(nb_classes, nb_hidden).normal_(0, epsilon)
b2 = torch.empty(nb_classes).normal_(0, epsilon)

#Weights,Biais,Activation = net.param() #à définir 
Weights= [w1,w2]
Biais = [b1,b2]
a = Tanh()
Activation = [a,a]
loss = LossMSE()

dl_dw = [torch.empty(w.size()) for w in Weights]
dl_db = [torch.empty(b.size()) for b in Biais]

#dl_dw1 = torch.empty(w1.size())
#dl_db1 = torch.empty(b1.size())
#dl_dw2 = torch.empty(w2.size())
#dl_db2 = torch.empty(b2.size())

for k in range(1000):

    # Back-prop

    acc_loss = 0
    nb_train_errors = 0
    
    for dw in dl_dw:
        dw.zero_()
    #dl_dw1.zero_()
    #dl_dw2.zero_()
    for db in dl_db:
        db.zero_()
    #dl_db1.zero_()
    #dl_db2.zero_()

    for n in range(nb_train_samples):
        input = [Weights,Biais,Activation, train_input[n]]
        x,s = forward(input)
        #print(len(x))
        gradwrtoutput = [Weights,Biais,Activation,train_target[n],[x,s],dl_dw,dl_db,loss]
        x2 = x[-1]
        pred = x2.max(0)[1].item()
        #print(pred)
        if train_target[n, pred] < 0.5:
            nb_train_errors = nb_train_errors + 1
        #print(x2.size())
        #print(train_target[n].size())
        acc_loss = acc_loss + loss.sigma(x2, train_target[n])
    
        dl_dw,dl_db=backward(gradwrtoutput)#(w1, b1, w2, b2,train_target[n],x0, s1, x1, s2, x2,dl_dw1, dl_db1, dl_dw2, dl_db2)
        #print(len(x))
    # Gradient step
    for i in range(len(Weights)):
        Weights[i]= Weights[i]-eta * dl_dw[i]
        Biais[i]= Biais[i]-eta * dl_db[i]
    #w1 = w1 - eta * dl_dw1
    #b1 = b1 - eta * dl_db1
    #w2 = w2 - eta * dl_dw2
    #b2 = b2 - eta * dl_db2


# In[ ]:




    # Test error

    nb_test_errors = 0

    for n in range(test_input.size(0)):
        _, _, _, _, x2 = forward_pass(w1, b1, w2, b2, test_input[n])

        pred = x2.max(0)[1].item()
        if test_target[n, pred] < 0.5: nb_test_errors = nb_test_errors + 1

    print('{:d} acc_train_loss {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'
          .format(k,
                  acc_loss,
                  (100 * nb_train_errors) / train_input.size(0),
                  (100 * nb_test_errors) / test_input.size(0)))


# In[ ]:


a=[[0,1,2,3],[1,2],[1,1,1,1]]
def f(input):
    a,b,c = input
    print(a)
    print(c)


# In[ ]:


f(a)


# In[ ]:


print(nb_train_samples)


# In[ ]:


a[len(a)]


# In[ ]:





# In[ ]:




