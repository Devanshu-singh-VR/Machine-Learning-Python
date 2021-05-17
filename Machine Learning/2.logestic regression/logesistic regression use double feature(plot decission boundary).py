import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt(open("D:\hello\ex2.txt","r"),delimiter=",")
x = data[:, 0:2]
y = data[:, 2]

x = np.array(x)
y=np.array(y)
k=[]
#theta
t = np.zeros(3)

#Plot data feature 1 vs feature 2 ('o' where y=0 and '+' where y=1)
y1 = np.argwhere(y==1)
y0 = np.argwhere(y==0)

plt.plot(x[y1, 0],x[y1, 1],linestyle='',marker='+',color='r')
plt.plot(x[y0, 0],x[y0, 1],linestyle='',marker='o',color='b')
plt.xlabel("feature 1")
plt.ylabel("featuer 2")
plt.title("hogaya wohooooo")

#modify x

x1 = data[:, 0]
x2 = data[:, 1]
for i in range(len(y)):
    k.append([1,x1[i],x2[i]])
x=k
x=np.array(x)

#assign values
m=len(y)


#compute

# sigmoid function
def sigmoid(z):
    ht = 1 / (1+np.exp(-z))
    return(ht)

# derivative of sigmoid
def d_sigmoid(z):
    return(sigmoid(z) * (1-sigmoid(z)))

#form cost function
def cost(t,x,y):
    k = x.dot(t.T)

    cost = (1.0 / m) * (-y.T.dot(np.log(sigmoid(k))) - (1 - y).T.dot(np.log(1 - sigmoid(k))))
    grad = (1.0/m) * (((sigmoid(k) - y) * d_sigmoid(k)).T.dot(x))
    return(cost,grad)

#take values

jval,gr = cost(t,x,y)
print("value of cost = ",jval)
print("value of grad = ",gr)

# using advanced optimisation
from scipy import optimize as opt

t, nfeval ,un = opt.fmin_tnc(func = cost,x0=t,args=(x,y))

new_cost,_ = cost(t,x,y)

print('new cost fx minimized val = ',new_cost)
print("optimized theta val = ",t)


#plot dcission boundary
x1=range(30,100)
x1 = np.array(x1)
x2 = (-t[0]-(t[1]*x1))/(t[2])
plt.plot(x1,x2)
plt.show()

def predict(t, x):

    p = sigmoid(x.dot(t.T)) >= 0.5
    return p.astype(int)
p = predict(t, x)
print 'Train Accuracy:', np.mean(p == y)
