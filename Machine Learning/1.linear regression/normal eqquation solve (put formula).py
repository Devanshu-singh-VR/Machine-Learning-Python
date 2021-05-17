import pandas as pd
import numpy as np

data = pd.read_excel("D:\hello\DSm.xlsx")
x = data.iloc[:, 0]
y = data.iloc[:, 2]
c = data.iloc[:, 1]

k=[]
for i in range(len(x)):
    k.append([1,x[i],c[i]])

x=k

x=np.array(x)
y=np.array(y)

print(x)

theta = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)

print(theta)
