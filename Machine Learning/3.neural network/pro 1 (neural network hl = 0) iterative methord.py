import numpy as np
import scipy as sp

#import data
x = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
y = np.array([[1],[0],[0],[1],[1]])

#define layers size
inputs = 3
outputs = 1

#sigmoid func
def sigmoid(k):
    return(1/(1+np.exp(-k)))

#derivative of sigmoid
def d_sigmoid(k):
    return((sigmoid(k)) * (1-sigmoid(k)))

#weights
w = np.random.rand(inputs,outputs)

alp = 0.1

#define cost func
for i in range(100000):

    #define forward
    z2 = x.dot(w)
    ht = sigmoid(z2)

    #cost
    j = (0.5/len(y)) *sum(ht - y)**2

    #gradient
    delta = np.multiply(ht-y , d_sigmoid(z2))
    d = x.T.dot(delta)

    w = w - (alp*d)

z2 = x.dot(w)
ht = sigmoid(z2)
print(ht)


