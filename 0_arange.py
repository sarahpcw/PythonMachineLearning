# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 21:38:00 2019

@author: u

#numpy.meshgrid(*xi, **kwargs)  #[source] #Return coordinate matrices from coordinate vectors.

#numpy.ravel(a, order='C') #[source] #Return a contiguous flattened array.

"""
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


print ( np.arange(0,10,2)) 
print ( np.arange(0,10,)) 
print ( np.arange(0,10,3)) 
print ( np.arange(0,20)) 
print ( np.arange(10,20)) 


#Now, we need to load the input data −
iris = datasets.load_iris()

#Now, we need to load the input data −
iris = datasets.load_iris()
#From this dataset, we are taking first two features as follows −
X = iris.data[:, :2]
y = iris.target
print (X)
print ("\n ----------------- y:: \n",y)

x_min = X[:, 0].min() - 1  # min of first col
x_max = X[:, 0].max() + 1
y_min = X[:, 1].min() - 1  # min of second col
y_max = X[:, 1].max() + 1 
h = (x_max / x_min)/100
print ("x min", "x max", "y min ", "y max ", "h")
print (x_min, x_max, y_min, y_max, h)

print ( np.arange(x_min, x_max, h) )
print ( np.arange(y_min, y_max, h) )
 #creates an array of integers between the 2 numbers , increment by the 3rd number
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  
# create all points onx and all points on y
# 
plt.scatter(xx,yy ); # s effects the dot size
plt.show()


z= np.linspace(0.3,-11.6,400) # Creates 400 numbers betweem 0.3 and -11.6
print (z.shape)
print (z)