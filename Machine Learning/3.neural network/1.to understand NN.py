import numpy as np
import scipy as sp

#import data
x = np.array(([1,3,5]) , dtype='float')
y = np.array(([0.66]) , dtype='float')

#define layers size
inputs = 3
outputs = 1

#sigmoid fnco
def sigmoid(k):
    return(1/(1+np.exp(-k)))

#derivative of sigmoid
def d_sigmoid(k):
    return(sigmoid(k) * (1-sigmoid(k)))

#weights
w1 = np.random.rand(inputs,outputs)

#define cost func
def cost(w1,x,y):

    #define forward
    z2 = x.dot(w1)
    ht = sigmoid(z2)

    #cost
    j = (0.5/len(y)) *sum(ht - y)**2

    #gradient
    d1 = (ht-y) * d_sigmoid(z2)
    grad = d1*x

    return(j,grad)

# using advanced optimisation
from scipy import optimize as opt

w1, nfeval ,un = opt.fmin_tnc(func = cost,x0=w1,args=(x,y))
print(w1)

#call answer
z2 = x.dot(w1)
ht = sigmoid(z2)
print(ht)