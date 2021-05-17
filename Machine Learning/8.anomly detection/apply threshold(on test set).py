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

#select threshold for X-val examples to test

def ming(p,y_val):
    k = (np.max(p) - np.min(p)) / 1000

    best_epsilon = 0
    best_f1 = 0

    for epsilon in np.arange(min(p), max(p), k):
        pridiction = p < epsilon

        tp = np.sum(pridiction[(y_val==True)])
        tn = np.sum(y_val[(pridiction==False)]==False)
        fp = np.sum(pridiction[(y_val==False)])
        fn = np.sum(y_val[(pridiction==False)]==True)
        if tp != 0:
            prec = 1.0 * tp / (tp + fp)
            rec = 1.0 * tp / (tp + fn)
            F1 = 2.0 * prec * rec / (prec + rec)
            if F1 > best_f1:
                best_f1 = F1
                best_epsilon = epsilon
    return(best_epsilon,best_f1)


p = anomly(X_val,me,sig)
epi,f1 = ming(p,y_val)
print("threshold = ",epi)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_val[:, 0], X_val[:, 1], p, marker='+',color='k')
plt.show()



