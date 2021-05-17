
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

def cost(t,x,y):


    j = (1.0/(2*m)) * np.sum(np.square(x.dot(t) - y))
    grad = (1.0/m) * x.T.dot(x.dot(t) - y)

    return(j,grad)

# using advanced optimisation
def train(x,t,y):

    t, nfeval ,un = opt.fmin_tnc(func = cost,x0=t,args=(x,y))
    return(t)

# call and give values to function

#form x as polynomial function
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

def error(X,y,X_val,y_val,X_test,y_test):

    e_val = np.zeros(5)
    e_train = np.zeros(5)

    for p in range(1,6):
        p = p
        x1,X1_val,t = power(X,X_val,p)

        t = train(x1,t,y)

        e_train[p-1] = (1.0/(2*m)) * np.sum(np.square(x1.dot(t) - y))
        e_val[p-1] = (1.0/(2*m)) * np.sum(np.square(X1_val.dot(t) - y_val))
    return(e_val,e_train)

e_val,e_train = error(X,y,X_val,y_val,X_test,y_test)
plt.plot(range(1,6),e_val,color='r')
plt.plot(range(1,6),e_train,color='b')
plt.xlabel('power')
plt.ylabel('error')
plt.show()














