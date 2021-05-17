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
    sigma = (1.0/m) * ((X-mean).T.dot(X-mean))

    return(mean,sigma)

def anomly(X,me,sig):
    m,n = X.shape
    a = len(me)
    Xm = X - me

    p = (1.0/(2*np.pi))**(a/0.5) * np.linalg.det(sig)**0.5 * np.exp(-np.sum(Xm.dot(np.linalg.pinv(sig)) * Xm , axis=1))

    return(p)

me,sig = gaussian(X)
prob = anomly(X,me,sig)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], prob, marker='+',color='k')
plt.show()






