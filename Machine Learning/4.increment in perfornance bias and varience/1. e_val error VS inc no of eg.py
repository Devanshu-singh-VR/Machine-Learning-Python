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

plt.figure()
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
x = np.hstack((np.ones((m, 1)), X))
t = np.zeros((2,1))
t = train(x,t,y)


#plot [e_cv , e_test ((VS)) number of training eg]
def error(t,x,y,X_val,y_val,X_test,y_test):

    e_val = np.zeros(len(y))
    e_train = np.zeros(len(y))

    for i in range(1,len(y)):
        t = train(x[0:i],t,y[0:i])
        e_train[i-1] = (1.0/(2*m)) * np.sum(np.square(x[0:i].dot(t) - y[0:i]))
        e_val[i-1] = (1.0/(2*m)) * np.sum(np.square(np.hstack((np.ones((len(X_val), 1)), X_val)).dot(t) - y_val))
    return(e_val,e_train)

e_val,e_train = error(t,x,y,X_val,y_val,X_test,y_test)

plt.plot(range(0,len(y)),e_val,color='r')
plt.plot(range(0,len(y)),e_train,color='b')
plt.show()














