import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


mat_data = sio.loadmat('D:\hello\ex7data1.mat')
X = mat_data['X']

def feature_normalize(X):

    mean = np.mean(X, axis=0)
    standard_dev = np.std(X, axis=0, ddof=1)
    X_norm = (X - mean) / standard_dev
    return(X_norm)

def pca(X):

    m, n = X.shape
    sigma = X.T.dot(X) / m
    U, S, V = np.linalg.svd(sigma)
    return(U, S, V)

a = feature_normalize(X)
u,s,v = pca(X)

Z = X.dot(u[:,0])
print("compressed data = ",Z)

#approx recover data
X_1 = Z*u[:1,0]
X_2 = Z*u[1:2,0]
X_recov = np.vstack((X_1,X_2)).T
print("approx recover data = ",X_recov)

plt.figure()
plt.xlabel("X1")
plt.ylabel("X2")
plt.scatter(X[:,0],X[:,1],color='b')
plt.scatter(X_recov[:,0],X_recov[:,1],color='r')
plt.legend(["inital data ","recover data "], loc='upper left', numpoints=2)
plt.show()




