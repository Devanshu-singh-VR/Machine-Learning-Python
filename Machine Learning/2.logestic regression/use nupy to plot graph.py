import matplotlib.pyplot as plt
import numpy as np
#plot dcission boundary
x=range(-100,100)
x=np.array(x)
y=x*3 + 2
plt.bar(x,y)
plt.show()