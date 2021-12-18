from torch import save
from torch import split
from torch import load
from torch import cat
from torch import randperm
import math
from torch import empty
from torch import set_grad_enabled
from torch import no_grad
import numpy as np
from time import time
import matplotlib.pyplot as plt

import NNmodule as nn
from NNmodule import MSE
from NNmodule import SGD
from NNmodule import Linear
from NNmodule import Tanh
from NNmodule import Relu
set_grad_enabled(False)



def one_hot(a):
    num_class = a.max()+1
    N=a.size(0)
    out = empty(N,num_class).zero_()
    for i in range(N):
        out[i][a[i]]=1 
    return out 
        




def generate_disc_set(nb):
    input = empty(nb, 2).uniform_(0, 1)
    target = input.sub(1/2).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).long()
    
    return input, one_hot(target)



nb_train_samples=1000
nb_validation_samples=1000
nb_epochs =1000 
nb_hidden = 25
mini_batch_size =250 
eta = 1e-2 / nb_train_samples
tol_grad = 1


train_input, train_target = generate_disc_set(nb_train_samples)
validation_input, validation_target = generate_disc_set(nb_validation_samples)

mean, std = train_input.mean(), train_input.std()
train_input.sub_(mean).div_(std)
validation_input.sub_(mean).div_(std)


nb_test_samples = 1000
test_input,test_target = generate_disc_set(nb_test_samples)
test_plot = test_input.clone()
test_input.sub_(mean).div_(std)



nb_classes = train_target.size(1)



net = nn.Sequential(Linear (train_input.size(1),(nb_hidden)),Relu(),
                 Linear( nb_hidden,nb_hidden),Tanh(),
                 Linear( nb_hidden,nb_hidden),Tanh(),
                 Linear( nb_hidden,nb_hidden),Relu(),
                 Linear( nb_hidden,nb_classes),Relu()).init_net()

# store inital weights for plot later
w0 = net.param()[0].param()[0]
b0 = net.param()[0].param()[1]
a0 = lambda x:  x.mul_(-w0[0]).sub_(b0).div_(w0[1])


x0 = empty(w0.size()).zero_()
x0[0].add_(-1)
x0[1].add_(1)
y0 = empty(w0.size()).zero_()
y0[0].add_(-1)
y0[1].add_(1)
y0=a0(y0) 

loss = nn.Loss(MSE(),net)
_,xi,_,_ = loss.predict(test_input,test_target)
ipred_class0 = (xi.argmax(1)-1).nonzero()
ipred_class1 = xi.argmax(1).nonzero()
iclass0 =(test_target.argmax(1)-1).nonzero()
iclass1 =(test_target.argmax(1)).nonzero()
combined = cat((ipred_class0, iclass0))
uniques, counts = combined.unique(return_counts=True)
imiss_classify0 = uniques[counts == 1]
iwell_classify0 = uniques[counts > 1]
combined = cat((ipred_class1, iclass1))
uniques, counts = combined.unique(return_counts=True)
imiss_classify1 = uniques[counts == 1]
iwell_classify1 = uniques[counts > 1]

combined = cat((imiss_classify0.view(-1), iclass0.view(-1)))
uniques, counts = combined.unique(return_counts=True)
imiss_classify0 = uniques[counts > 1]

combined = cat((imiss_classify1.view(-1), iclass1.view(-1)))
uniques, counts = combined.unique(return_counts=True)
imiss_classify1 = uniques[counts > 1]


#### start training ####

optimizer= SGD(net,eta)


validation_error = []
train_error = []
train_acc = []


e=0
list_grad_norm = []
grad_norm=100000
zeit = []
while (e<nb_epochs)&(grad_norm>tol_grad):
    i=0
    loss.nb_train_errors  = 0
    loss.acc_loss=0
    train_input,train_target=split(cat((train_input,train_target),1)[randperm(nb_train_samples)],train_input.size(1),1)

    for b in range(0, train_input.size(0), mini_batch_size):

        net.assign(train_input.narrow(0, b, mini_batch_size),train_target.narrow(0, b, mini_batch_size))
        tic = time()
    	# Back-prop
        loss.backward()#This function call the forward method of the network then the backward for calculating the accumulators
        toc = time()
        zeit.append((toc-tic))
        i+=1
    # Gradient step with SGD
    optimizer.step()
    net.zero_grad()
    
    # validation error
    grad_norm = optimizer.get_grad_norm()#/nb_train_samples
    list_grad_norm.append(grad_norm)
    _,_,nb_test_errors,_=loss.predict(validation_input,validation_target)   
    print('{:d} acc_train_loss {:.02f} acc_train_error {:.02f}% validation_error {:.02f}% grad_norm {:.05f}'
          .format(e,
                  loss.acc_loss,
                  (100 * loss.nb_train_errors) / nb_train_samples,
                  (100 * nb_test_errors) / nb_validation_samples,grad_norm))
   
    validation_error.append((100 * nb_test_errors) / validation_input.size(0))
    train_error.append((100 * loss.nb_train_errors) / train_input.size(0))
    train_acc.append(loss.acc_loss)
    e+=1
    



	# plot errors 

plt.figure(0)
plt.semilogy(train_acc,label='train_acc')
plt.semilogy(list_grad_norm,label='grad_norm')
plt.legend()
plt.show()

plt.figure(1)
plt.semilogy(validation_error,label='validation_error')
plt.semilogy(train_error,label='train_error')
plt.legend()
plt.show()




save(net,'./project_net_hybrid')


# test error


_,x,nb_test_errors,test_acc=loss.predict(test_input,test_target)   

print('\nacc_train_error {:.02f}% acc_test_error {:.02f}%'
          .format(
                  (100 * loss.nb_train_errors) / nb_train_samples,
                  (100 * nb_test_errors) / nb_test_samples))
                  

	#plot representation graphique

pred_class0 = (x.argmax(1)-1).nonzero()        
pred_class1 = x.argmax(1).nonzero()
class0 =(test_target.argmax(1)-1).nonzero()
class1 =(test_target.argmax(1)).nonzero()
combined = cat((pred_class0, class0))
uniques, counts = combined.unique(return_counts=True)
miss_classify0 = uniques[counts == 1]
well_classify0 = uniques[counts > 1]
combined = cat((pred_class1, class1))
uniques, counts = combined.unique(return_counts=True)
miss_classify1 = uniques[counts == 1]
well_classify1 = uniques[counts > 1]

combined = cat((miss_classify0.view(-1), class0.view(-1)))
uniques, counts = combined.unique(return_counts=True)
miss_classify0 = uniques[counts > 1]

combined = cat((miss_classify1.view(-1), class1.view(-1)))
uniques, counts = combined.unique(return_counts=True)
miss_classify1 = uniques[counts > 1]



pred_class0 = (x.argmax(1)-1).nonzero()        
pred_class1 = x.argmax(1).nonzero()
class0 =(test_target.argmax(1)-1).nonzero().view(-1)
class1 =(test_target.argmax(1)).nonzero().view(-1)


fig , (ax1,ax2)=plt.subplots(1,2)
ax1.set_title('initial classification')
ax1.set(xlim=(-.2, 1.2), ylim=(-.2, 1.2))
combined = cat((miss_classify1.view(-1), class1.view(-1)))
uniques, counts = combined.unique(return_counts=True)
miss_classify1 = uniques[counts > 1]
#ax1.scatter(test_plot[class1].t()[0],test_plot[class1].t()[1],c='red')
#ax1.scatter(test_plot[class0].t()[0],test_plot[class0].t()[1],c='blue')
ax1.scatter(test_plot[iwell_classify1].t()[0],test_plot[iwell_classify1].t()[1],c='red')
ax1.scatter(test_plot[iwell_classify0].t()[0],test_plot[iwell_classify0].t()[1],c='blue')
ax1.scatter(test_plot[imiss_classify0].t()[0],test_plot[imiss_classify0].t()[1],c='cyan',label='blue_misclassified')
ax1.scatter(test_plot[imiss_classify1].t()[0],test_plot[imiss_classify1].t()[1],c='yellow',label='red_misclassified')
ax1.legend()

ax2.set(xlim=(-.2, 1.2), ylim=(-.2, 1.2))
ax2.set_title('Classification after training')
ax2.scatter(test_plot[well_classify1].t()[0],test_plot[well_classify1].t()[1],c='red')
ax2.scatter(test_plot[well_classify0].t()[0],test_plot[well_classify0].t()[1],c='blue')
ax2.scatter(test_plot[miss_classify0].t()[0],test_plot[miss_classify0].t()[1],c='cyan',label='blue_misclassified')
ax2.scatter(test_plot[miss_classify1].t()[0],test_plot[miss_classify1].t()[1],c='yellow',label='red_misclassified')

w = net.param()[0].param()[0]
b = net.param()[0].param()[1]
a = lambda x:  x.mul_(-w[0]).sub_(b).div_(w[1])

marker_style = dict(linestyle='-', color='0.8', markersize=10,
                    markerfacecolor="tab:blue", markeredgecolor="tab:blue")
markers=['$'+str(i)+'$' for i in range(w.size(1))]
marker_style.update(markeredgecolor="k", markersize=5)


x = empty(w.size()).zero_()
x[0].add_(-1)
x[1].add_(1)
y = empty(w.size()).zero_()
y[0].add_(-1)
y[1].add_(1)
y=a(y) 
for i in range(w.size(1)):
	ax1.plot(x0.div(2).add(0.5).t()[i],y0.div(2).add(0.5).t()[i], marker=markers[i], **marker_style)
	ax2.plot(x.div(2).add(0.5).t()[i],y.div(2).add(0.5).t()[i], marker=markers[i], **marker_style)
#format_axes(ax)
ax2.legend()
plt.show()

from numpy import array
zeit=array(zeit)
print('\nmoyenne des temps par backward : ',zeit.mean())
print('\ntemps d entrainement totale : ', zeit.sum(),'\n')
