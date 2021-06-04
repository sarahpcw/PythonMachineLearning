# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 20:59:07 2021

@author: u
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
########## meshgrid

h = (400 / 3)/100
print (h)
xx = np.arange(1, 100, h)
yy = np.arange(101,200, h)
plt.scatter(xx,yy ); # s effects the dot size normally nothing or 20 or so
plt.show()


xx, yy =np.arange(3, 400, h), np.arange(103, 500, h)   # h is the increment
plt.scatter(xx,yy , s = 500); # s effects the dot size normally nothing or 20 or so
plt.show()

xx, yy = np.meshgrid(np.arange(3, 400, h), np.arange(103, 500, h))   # h is the increment
plt.scatter(xx,yy , s = 500); # s effects the dot size normally nothing or 20 or so
plt.show()


h = (400 / 3)/100
print (h)
xx, yy = np.meshgrid(np.arange(3, 400, h), np.arange(103, 500, h))   # h is the increment
plt.scatter(xx,yy , s = 500); # s effects the dot size normally nothing or 20 or so
plt.show()


iris = datasets.load_iris()#Now, we need to load the input data −
#From this dataset, we are taking first two features as follows −
X = iris.data[:, :2]
y = iris.target
#print (X)
#print ("\n ----------------- y:: \n",y)

x_min = X[:, 0].min() - 1  # min of first col
x_max = X[:, 0].max() + 1
y_min = X[:, 1].min() - 1  # min of second col
y_max = X[:, 1].max() + 1 
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  
# create all points onx and all points on y
plt.scatter(xx,yy , s = 20); # s effects the dot size normally nothing or 20 or so
plt.show()

