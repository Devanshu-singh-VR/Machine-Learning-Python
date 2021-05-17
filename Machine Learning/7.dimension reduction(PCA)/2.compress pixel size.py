import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


mat_data = sio.loadmat('D:\hello\ex7faces.mat')
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


Z = X.dot(u[:,0:308])

print('initial pixel size = ',X.shape)
print('compressed size = ',Z.shape)

#take value 308
for i in range(len(s)):
    a = sum(s[:i])/(sum(s) *1.0)
    print i+1," = ",a

