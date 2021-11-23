#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 


# In[2]:


class Module(object):
    def forward(self, *input):
        raise NotImplementedError
    def backward(self, *gradwrtoutput): 
        raise NotImplementedError
    def param(self): 
        return [] # c'est mieux de faire avec une liste ou un dicco ? pour pouvoir faire les share weight


# In[3]:


class Linear(Module):
    def __init__(self,input_size, hidden_size):
        super().__init__()
        self.p = []   
        self.p.append(torch.empty(input_size, hidden_size).normal_())#Weight
        self.p.append( torch.zeros(hidden_size)) #bias
    def param(self):
        return self.p
    def forward(self,x):
        return torch.mm(x,self.p[0])+self.p[1] #faire le resize de x à (1,input_size) ? 
    def backward(x):
        return torch.ones(input_size).mm(self.p[0])


# In[4]:


Linear(3,2).forward(torch.ones(1,3))


# In[5]:


class Tanh(Module):
    def __init__(self,*input):
        super().__init__()
        self.p= []
    def forward(self,x):
        return torch.tanh(x)
    def backward(self,x):
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)
    def param(self):
        return self.p


# In[6]:


Tanh().forward(2*torch.ones(1))


# In[7]:


class LossMSE(Module): #le loss doit il appartenir au module ? 
    def __init__(self):
        super().__init__()
        self.p = [] 
    def loss(self,x, y):
        return (self.forward(x) - y).pow(2).sum() # normalemnt le loss est une fonction allant tj vers les reelles ?
    def grad(self,x,y):
        W=self.param()
        return 2 * (x - y).sum()


# In[8]:


class Sequential(Module):
    def __init__(self, *input): 
        super().__init__() 
        self.layer = input
        self.p=[]
    def forward(self,x):
        for l in self.layer:
            x=l.forward(x)
        return x
        
    def backward(self,*gradwrtoutput):
        for l in range(len(self.layer),0,-1):
             0
        return 0
    def param(self):
        for l in self.layer:
            self.p.append(l.param())
        return self.p
        
        


# In[9]:


x=torch.ones((1,2))
a=Sequential(Linear(2,2),Tanh())
a.forward(x)


# In[10]:


torch.empty((2,3,4)).normal_().sum(0)


# In[11]:


a.param()


# In[ ]:




