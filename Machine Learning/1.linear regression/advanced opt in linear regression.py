import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


mat_data = sio.loadmat('D:\hello\ex5data1.mat')
X = mat_data['X']
y = mat_data['y'].ravel()


plt.figure()
plt.plot(X, y, linestyle='', marker='x', color='r')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

m=len(y)

def cost(t,x,y):

    j = (1.0/(2*m)) * np.sum(np.square(x.dot(t) - y))
    grad = (1.0/m) * x.T.dot(x.dot(t) - y)

    return(j,grad)

# using advanced optimisation
from scipy import optimize as opt
x = np.hstack((np.ones((m, 1)), X))
t = np.zeros((2,1))

t, nfeval ,un = opt.fmin_tnc(func = cost,x0=t,args=(x,y))

ht = t[0] + t[1]*X
plt.plot(X,ht,color='b')
plt.show()