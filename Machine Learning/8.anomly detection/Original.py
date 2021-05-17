import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mat_data = sio.loadmat('D:\hello\ex8data1.mat')
X = mat_data['X']
X_val = mat_data['Xval']
y_val = mat_data['yval'].ravel()

def gaussian(X):
    m,n = X.shape

    mean = np.mean(X,axis=0)
    sigma = np.var(X,axis=0)

    return(mean,sigma)

def anomly(X,me,sig):
    m,n = X.shape

    p = 1
    for i in range(n):
        p = p * (1.0/(2*np.pi*sig[i]))**(0.5) * np.exp(-(np.square(X[:,i] - me[i])/sig[i]))

    return(p)

me,sig = gaussian(X)
p = anomly(X,me,sig)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], p,color='m')
plt.show()

