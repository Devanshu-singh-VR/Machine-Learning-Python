import scipy as sp
import math as mt
import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt(open("D:\hello\lorcic.txt","r"),delimiter=",")
x = data[:, 0:2]
y = data[:, 2]

x = np.array(x)
y=np.array(y)
k=[]

#Plot data feature 1 vs feature 2 ('o' where y=0 and '+' where y=1)
y1 = np.argwhere(y==1)
y0 = np.argwhere(y==0)

plt.plot(x[y1, 0],x[y1, 1],linestyle='',marker='+',color='y')
plt.plot(x[y0, 0],x[y0, 1],linestyle='',marker='o',color='b')
plt.xlabel("feature 2")
plt.ylabel("featuer 1")
plt.show()
# map features due to non linear graph (i was thinking may be it will be an ellips)
def map_feature(x1, x2):

    degree = 2
    Out = np.ones(len(x1))

    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            tmp = np.power(x1, i - j) * np.power(x2, j)
            Out = np.vstack((Out, tmp))
    return Out.T

x = map_feature(x[:, 0], x[:, 1])

m, n = x.shape
# Initialize fitting parameters
t = np.zeros(n)
# Set regularization parameter lambda to 1
l = 1.0

#compute

# sigmoid function
def sigmoid(z):
    ht = 1 / (1+np.exp(-z))
    return(ht)

# derivative of sigmoid
def d_sigmoid(z):
    return(sigmoid(z) * (1-sigmoid(z)))

#lambda
l = 0.0001

# cost function for regularization
def cost(t,x,y):
    mt = np.eye(len(t),dtype='float')
    reg = mt.dot(t)

    k = x.dot(t.T)

    cost = ((1.0 / m) * (-y.T.dot(np.log(sigmoid(k))) - (1 - y).T.dot(np.log(1 - sigmoid(k))))) + (1.0/m)*l* sum(reg**2)
    grad = (1.0/m) * (((sigmoid(k) - y ) * d_sigmoid(k)).T.dot(x)) + (1.0/m)*l* (reg)
    return(cost,grad)

#take values

jval,gr = cost(t,x,y)
print("value of cost = ",jval)
print("value of grad = ",gr)

# using advanced optimisation
from scipy import optimize as opt

t, nfeval, rc = opt.fmin_tnc(func = cost,x0=t,args=(x,y))

new_cost,_ = cost(t,x,y)

print('new cost fx minimized val = ',new_cost)
print("optimized theta val = ",t)

def predict(t, x):

    p = sigmoid(x.dot(t.T)) >= 0.5
    return p.astype(int)
p = predict(t, x)
print 'Train Accuracy:', np.mean(p == y) * 100

#check
x1 = float(input('second feature x1 = '))
x2 = float(input('first feature x2 = '))

X = np.array([1.0,x1,x2,x1**2,x1*x2,x2**2])
z = X.dot(t.T)
def sigmoid(z):
    ht = 1 / (1+np.exp(-z))
    return(ht)
if sigmoid(z)>0.5:
    print(1)
else:
    print(0)
x = data[:, 0:2]
y = data[:, 2]

x = np.array(x)
y=np.array(y)
k=[]

#Plot data feature 1 vs feature 2 ('o' where y=0 and '+' where y=1)
y1 = np.argwhere(y==1)
y0 = np.argwhere(y==0)

plt.plot(x[y1, 0],x[y1, 1],linestyle='',marker='+',color='y')
plt.plot(x[y0, 0],x[y0, 1],linestyle='',marker='o',color='b')
plt.plot(x2,x1,linestyle='',marker='D',color='r')
plt.xlabel("feature 2")
plt.ylabel("featuer 1")
plt.show()

p =input('press enter to exit')




