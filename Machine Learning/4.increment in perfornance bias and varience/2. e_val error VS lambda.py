
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import optimize as opt

mat_data = sio.loadmat('D:\hello\ex5data1.mat')
X = mat_data['X']
y = mat_data['y'].ravel()
X_test = mat_data['Xtest']
y_test = mat_data['ytest'].ravel()
X_val = mat_data['Xval']
y_val = mat_data['yval'].ravel()
m = X.shape[0]
m_val = X_val.shape[0]
m_test = X_test.shape[0]

#plt.figure()
#plt.plot(X, y, linestyle='', marker='x', color='r')
#plt.xlabel('Change in water level (x)')
#plt.ylabel('Water flowing out of the dam (y)')

# train Theta
m=len(y)

def cost(t,x,y,L):

    j = (1.0/(2*m)) * np.sum(np.square(x.dot(t) - y)) + 1.0 * L / (2 * m) * np.sum(np.square(t[1:]))
    grad = (1.0/m) * x.T.dot(x.dot(t) - y) + 1.0 * (L / m) * np.sum(t[1:])

    return(j,grad)

# using advanced optimisation
def train(x,t,y,L):

    t, nfeval ,un = opt.fmin_tnc(func = cost,x0=t,args=(x,y,L))
    return(t)

# call and give values to function

#form x as polynomial function
p = 11
def power(X,X_val,p):
    s1 = np.ones((len(y),1))
    s2 = np.ones((len(y_val),1))

    for i in range(0,p):
        g1 = np.power(X,i+1)
        s1 = np.hstack((s1 , g1))
        g2 = np.power(X_val,i+1)
        s2 = np.hstack((s2 , g2))

    t = np.zeros((p+1,1))
    return(s1,s2,t)

x,X_val,t = power(X,X_val,p)

def error(t,x,y,X_val,y_val):
    L = 0
    l = []
    e_val = np.zeros(9)
    e_train = np.zeros(9)
    for k in range(0,9):
        L = L + 0.1
        l.append(L)

        t = train(x,t,y,L)

        e_train[k] = (1.0/(2*m)) * np.sum(np.square(x.dot(t) - y))
        e_val[k] = (1.0/(2*m)) * np.sum(np.square(X_val.dot(t) - y_val))
    return(e_val,e_train,l)

e_val,e_train,l = error(t,x,y,X_val,y_val)

plt.plot(l,e_val,color='r')
plt.plot(l,e_train,color='b')
plt.show()















