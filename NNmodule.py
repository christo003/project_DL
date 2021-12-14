#!/usr/bin/env python
# coding: utf-8


import math
from torch import empty
from torch import set_grad_enabled
from torch import no_grad
import numpy as np
set_grad_enabled(False)


class Module(object):
    
    def forward(self, *input):
        raise NotImplementedError
    def backward(self, *gradwrtoutput): 
        raise NotImplementedError
    def param(self): 
        return []
    def xavier_normal_(self,tensor, fan_in , fan_out,gain = 1):
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        with no_grad():
            return tensor.normal_(0, std)
    def xavier_uniform_(self,tensor, fan_in , fan_out,gain = 1):
        a = gain * math.sqrt(6.0 / (fan_in + fan_out))
        with no_grad():
            return tensor.uniform_(-a, a)





class Parameters(Module):
    """
    Implement the fully connected layer module
    """
    def __init__(self,linear,gain=5.0/3):
        super().__init__()

        self.hidden_size = linear.hidden_size
        self.input_size = linear.input_size
        # Initialisation with Xavier methods
        self.weights=self.xavier_normal_(empty(self.input_size, self.hidden_size),self.input_size,self.hidden_size,gain)#.uniform_(-1/math.sqrt(input_size),1/math.sqrt(input_size))#Weight
        self.biais=self.xavier_normal_(empty(self.hidden_size),self.input_size,self.hidden_size,gain)#.uniform_(-1/math.sqrt(input_size),1/math.sqrt(input_size))#bias

        
    
    def sigma(self,x):
        out = 0
        if len(x.size())>1:#Processing for mini-batch
            one_matrix = empty(x.size(0),self.hidden_size).zero_().add(1) #we multiply the result with the batch size
            out = x.mm(self.weights)+self.biais.view(1,-1)*one_matrix #Could be optimize with broadcasting
        else : #single input
            #W*x + b
            out = self.weights.t().mv(x)+self.biais 
        return out 
    def dsigma(self,x):
        out = 0
        if len(x.size())>1:
            out = x.mm(self.weights.T)#matrix output (X (M_batch x N_input_size) * W (N input_size x hidden_size) = out (M_batch x hidden_size))
        else : 
            out = self.weights().mv(x)#vector output
        return out
    def param(self):
        return [self.weights,self.biais]
    def set_param(self,new_w,new_b):
        """
        Allow to update the parameters when we have done the optimize step calculation.
        """
        self.weights= new_w
        self.biais = new_b
    def backward(self, *gradwrtoutput):
        """"
        l(s2(a2(s1(a1(x)))))' = l'(s2(a2(s1(x))))(s2(a2(s1(x))))' = l'(s2(a2(s1(x))))(s2'(a2(s1(x))))a2'(s1(x))
        = l'(s2(a2(s1(x))))(s2'(a2(s1(x))))a2'(s1(x))s1'(a1(x))
        
        où s1(x) = s1(W_{n-1}( ... s_{1}( W_{1} x + b_{1} )+b_{n-2} )+ b_{n-1}) 
        
        dl_ds = l'(s(a(x)))
        
        ds_da = s'(a(x))
        
        dl_ds = l'(s(a(x)))(s'(a(x)))
        
        dl_dw = dl_ds * da_dw 
         
        dl_db = dl_ds * da_db
        
        """
        dl_ds = gradwrtoutput[0]
        
        #da_dw contient N element de H size 
        #dl_dw la variation de la perte par rapport au poids,
        #pour la données j du batch, 
        #dl_dw = [(da_dw)_j1 * dl_ds ,(da_dw)_j2 * dl_ds,...,(da_dw)_jh * dl_ds]
        #où dl_ds est un vecteur 
        
        
        dl_dx = self.dsigma(dl_ds)
        
        return dl_dx
        
    def forward(self, *input):
        return self.sigma(*input)
    
    
    def zero_grad(self):
        self.grad_w.zero_()
        self.grad_b.zero_()
        
  
        
    def get_grad(self):
        return [self.grad_w,self.grad_b]



class Linear(Module):
    """
    Implement the fully connected layer module
    """
    def __init__(self,input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
  


# ### Activation Functions



class Activation(Module):
    """
    Implement activation layer tanh
    """
    def __init__(self,activation):
        super().__init__()
        self.ds_da=0
        self.s=0
        self.da_dw = 0
        self.sigma =activation.sigma
        if hasattr(activation,'dsigma') :
            self.dsigma=activation.dsigma
            
        else:  
            0
            #chercher la dérivé dsigma 
    
    
    def backward(self, *gradwrtoutput):
        """"
        l(s(w(x)))' = l'(s(g(x)))(s(g(x)))' = l'(s(g(x)))(s'(g(x)))g'(x)
        
        où a(x) = W_{n} (s_{n-1} (W_{n-1}( ... s_{1}( W_{1} x + b_{1} )+b_{n-2} )+ b_{n-1} ) + b_{n}
        
        dl_dx = l'(s(a(x)))
        
        ds_da = s'(a(x))
        
        dl_ds = l'(s(a(x)))(s'(a(x)))
        
        
        """
        
        dl_dx = gradwrtoutput[0]
        da_dw = self.da_dw
        #1ere itération : récupéeration de la dérivée du loss evalué à l'output du net
        #autre :  
    
        dl_ds = self.ds_da*dl_dx #avec dérivé évalué au point du forward 
        dl_dw =( da_dw.view(da_dw.size(0),da_dw.size(1),1).matmul(dl_ds.view(dl_ds.size(0),1,dl_ds.size(1))))
       
        
        return dl_ds,dl_dw
    def forward(self, *input):  
        self.da_dw,self.s = input
        self.ds_da = self.dsigma(self.s)#calcule de la dérivé évalué au point du forward qu'on stock pour le backward
        return self.sigma(self.s)




class Tanh( Module ) :
    def __init__(self):
        super().__init__()
        self.ds_da = 0
    def sigma(self,x):
        return x.tanh()
    def dsigma(self,x):
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)




class Relu( Module ) :
    def __init__(self):
        super().__init__()
        self.ds_da = 0
    def sigma(self,x):
        return x.max(empty(x.size()).zero_())
    def dsigma(self,x):
        out = empty(x.size()).zero_()
        out[x>0]=1
        return out
    


# ### Loss functions



class Loss(Module):
    """
    Loss has a network and a loss. 
    It uses the "sigma" method of the loss function (MSE)
    """
    def __init__(self,loss,net):
        super().__init__()
        self.net = net
        self.loss = loss
        self.acc_loss=0
        self.nb_train_errors=0
        
    def sigma(self,x,y):
        return self.loss.sigma(x,y)
    
    def dsigma(self,x,y):
        return self.dloss.sigma(x,y)
    
    def assign(self, net):
        self.net = net 
    
    def predict(self,x,test_target):
        """"
        just to evaluate the prediction of the network on one mini-batch
        """
        
        return (test_target.argmax(1)!=x.argmax(1)).sum(),self.loss.sigma(x, test_target).sum()
            
        
    def backward(self,*gradwrtoutput):
        #This function call the forward method of the network for one mini-batch, and then the backward function of the network
        x = self.net.forward(self.net.train)
        
        #x has row of mini_batch size and column of linear output (before activation))
        #s has row of mini_batch size and column of activation function (after activation))   
        nb_train_errors,acc_loss=self.predict(x,self.net.train_target)
        self.nb_train_errors+=nb_train_errors
        self.acc_loss+=acc_loss
        dl_dx2 = self.loss.dsigma(x, self.net.train_target)
        self.net.backward(dl_dx2)





class MSE(Module):
    """
    Implement the loss function Mean Square Error
    """
    def __init__(self):
        super().__init__()       
    def sigma(self,x,y):      
        return (x - y).pow(2).sum()
    def dsigma(self,x,y):   
        return 2*(x - y)
    


# ### Optimizer



class SGD(Module):
    def __init__(self,net,lr):
        super().__init__()
        self.net = net
        self.lr = lr
    def sigma(self,param,grad):
        out=[]
        for i in range(len(param)):
            out.append(param[i] - self.lr * grad[i].sum(0) ) #en faisant que n(t,b) sot séquentielle
        return out
    def step(self):
        for i in range(len(self.net.param())):
            new_w,new_b=self.sigma(self.net.get_param(i),self.net.get_grad(i))
            self.net.set_param(i,new_w,new_b)



# ### net module



class Net(Module):
    def __init__(self):
        super().__init__()
        self.Parameters = [] #List of fully connected layers
        self.Activation = [] #List of activation functions
        self.dl_dw=0
        self.dl_db=0
        self.train = []
        self.train_target = []
        self.num_sample = -1
        self.num_parameters = 0
    def forward(self,*input):
        x=input[0]
        for i in range(len(self.Activation)):
            s = self.Parameters[i].forward(x)
            x = self.Activation[i].forward(x,s)
        return x

    def backward(self,*gradwrtoutput):
        """
        call by the Loss module for updating the weigths of the network
        """
        N=self.num_parameters
        dl_dx = gradwrtoutput[0]
        dl_ds,dl_dw = self.Activation[-1].backward(dl_dx)
        self.dl_dw[-1].add_(dl_dw)
        self.dl_db[-1].add_(dl_ds)

        for i in range(N-1,0,-1): #Backpropagate for each layer once we took care of the loss
            dl_dx = self.Parameters[i].backward(dl_ds)
            dl_ds,dl_dw = self.Activation[i-1].backward(dl_dx)
            self.dl_dw[i-1].add_(dl_dw)
            self.dl_db[i-1].add_(dl_ds)
            
            
            
    def param(self):
        return self.Parameters
    
    def init(self,new_Parameters,new_Activation):
        self.num_parameters= len(new_Parameters)
        self.Parameters= new_Parameters
        self.Activation = new_Activation

    def set_param(self,num_layer,new_w,new_b):
        self.Parameters[num_layer].set_param(new_w,new_b)
        
    def get_grad(self,num_layer):
        return [self.dl_dw[num_layer],self.dl_db[num_layer]]
    
    def get_param(self,num_layer):
        return self.Parameters[num_layer].param()
    
    def zero_grad(self):
        for dw in self.dl_dw:
            dw.zero_()
        for db in self.dl_db:
            db.zero_()
        

    def assign(self, train,train_target):
        
        if len(train.size())>1: #mini batch
            self.train = train
            self.train_target = train_target
            if train.size(0)!=self.num_sample:
                self.num_sample = train.size(0)
                self.dl_dw = [empty(self.num_sample,p.param()[0].size(0),p.param()[0].size(1)).zero_() for p in self.Parameters]
                self.dl_db = [empty(self.num_sample,p.param()[1].size(0)).zero_() for p in self.Parameters]
                   
        else : 
            self.num_sample = 1
            self.train = train
            self.train_target = train_target
           

        



class Sequential(Module):
    def __init__(self,*layers):
        super().__init__()
        self.layers = layers 
    def init_net(self):
        net = Net()
        new_Parameters=[]
        new_Activation=[]
        i = 0
        for k in range(len(self.layers)-1,-1,-1) : 
            if np.mod(i,2)==0:
                if type(self.layers[k])==type(Tanh()):
                    gain=5/3
                if type(self.layers[k])==type(Relu()):
                    gain=math.sqrt(2.0)
                new_Activation.insert(0,Activation(self.layers[k]))
                
            else : 
                
                new_Parameters.insert(0,Parameters(self.layers[k],gain))
            i=i+1
        net.init(new_Parameters,new_Activation)
        return net 




