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
    input = empty(nb, 2).uniform_(-1, 1)
    target = input.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2).long()
    
    return input, one_hot(target)






nb_train_samples=1000
nb_test_samples=1000
nb_epochs = 1000
nb_hidden = 25
mini_batch_size = 100
eta = 1e-1 / nb_train_samples
epsilon = 1e-6


train_input, train_target = generate_disc_set(nb_train_samples)
test_input, test_target = generate_disc_set(nb_test_samples)

mean, std = train_input.mean(), train_input.std()

train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)


nb_classes = train_target.size(1)



net = nn.Sequential(Linear (train_input.size(1),(nb_hidden)),Relu(),
                 Linear( nb_hidden,nb_hidden),Tanh(),
                 Linear( nb_hidden,nb_hidden),Relu(),
                 Linear( nb_hidden,nb_hidden),Relu(),
                 Linear( nb_hidden,nb_classes),Tanh()).init_net()

loss = nn.Loss(MSE(),net)
optimizer= SGD(net,eta)


test_error = empty(nb_epochs).zero_()
train_error = empty(nb_epochs).zero_()
train_acc = empty(nb_epochs).zero_()



zeit = empty(nb_epochs,int(nb_train_samples/mini_batch_size)).zero_()
for e in range(nb_epochs):
    i=0
    loss.nb_train_errors  = 0
    loss.acc_loss=0

    for b in range(0, train_input.size(0), mini_batch_size):



        net.assign(train_input.narrow(0, b, mini_batch_size),train_target.narrow(0, b, mini_batch_size))
        tic = time()
    	# Back-prop
        loss.backward()#This function call the forward method of the network then the backward for calculating the accumulators
        toc = time()
        zeit[e][i]=(toc-tic)
        i+=1
        # Gradient step with SGD
        optimizer.step()
    net.zero_grad()
    
    # Test error
    nb_test_errors,_=loss.predict(net.forward(test_input),test_target)   
    print('{:d} acc_train_loss {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'
          .format(e,
                  loss.acc_loss.log(),
                  (100 * loss.nb_train_errors) / train_input.size(0),
                  (100 * nb_test_errors) / test_input.size(0)))
   
    test_error[e]=(100 * nb_test_errors) / test_input.size(0)
    train_error[e]=(100 * loss.nb_train_errors) / train_input.size(0)
    train_acc[e]=loss.acc_loss.log()




	# plot errors 
plt.figure()
plt.semilogy(test_error,label='test_error')
plt.semilogy(train_error,label='train_error')
plt.legend()
plt.show()
plt.figure()
plt.semilogy(train_acc,label='train_acc')
plt.legend()
plt.show()



from torch import save
from torch import load
from torch import cat
save(net,'./project_net_hybrid')



	#plot representation graphique
x = net.forward(test_input)
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

plt.figure()
plt.scatter(test_input[well_classify1].t()[0],test_input[well_classify1].t()[1],c='red')
plt.scatter(test_input[well_classify0].t()[0],test_input[well_classify0].t()[1],c='blue')
plt.scatter(test_input[miss_classify0].t()[0],test_input[miss_classify0].t()[1],c='cyan',label='blue_misclassified')
plt.scatter(test_input[miss_classify1].t()[0],test_input[miss_classify1].t()[1],c='yellow',label='red_misclassified')
plt.legend()
plt.show()



x = net.forward(test_input)
pred_class0 = (x.argmax(1)-1).nonzero()        
pred_class1 = x.argmax(1).nonzero()
class0 =(test_target.argmax(1)-1).nonzero().view(-1)
class1 =(test_target.argmax(1)).nonzero().view(-1)

plt.figure()
combined = cat((miss_classify1.view(-1), class1.view(-1)))
uniques, counts = combined.unique(return_counts=True)
miss_classify1 = uniques[counts > 1]
plt.scatter(test_input[class1].t()[0],test_input[class1].t()[1],c='red')
plt.scatter(test_input[class0].t()[0],test_input[class0].t()[1],c='blue')
plt.show()



print('moyenne des temps par backward : ',zeit.mean())
print('temps d entrainement totale : ', zeit.sum())
