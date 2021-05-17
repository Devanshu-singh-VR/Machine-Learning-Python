import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import svm

# Load from ex6data1
mat_data = sio.loadmat('D:\hello\ex3data1.mat')
X = mat_data['X']
y = mat_data['y'].ravel()

# Change the C value below and see how the decision boundary varies (e.g., try C = 1000).
clf = svm.SVC(kernel = 'rbf',gamma='scale',C=1)
clf.fit(X,y)
a = clf.predict(np.array([X[2000,:]]))
print(a)