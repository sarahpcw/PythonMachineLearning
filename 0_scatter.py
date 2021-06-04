# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 23:21:44 2019

@author: u

Scatter plots

"""

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

z= np.linspace(0.3,-11.6,400)
print (z.shape)
print (z)

h = (400 / 3)/100
print (h)
xx = np.arange(1, 100, h)
yy = np.arange(101,200, h)
plt.scatter(xx,yy ); # s effects the dot size normally nothing or 20 or so
plt.show()

 

xx, yy = np.meshgrid(np.arange(3, 400, h), np.arange(103, 500, h))   # h is the increment
xx, yy =np.arange(3, 400, h), np.arange(103, 500, h)   # h is the increment
plt.scatter(xx,yy , s = 500); # s effects the dot size normally nothing or 20 or so
plt.show()



 
####################### 2d  scatter
#The following code will generate the 2D, containing four blobs −
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples = 400, centers = 6, cluster_std = 0.60, random_state = 1) #cluster_std :The standard deviation of the clusters.
print ('X.shape',X.shape) 
print ('X[:, 0].max()', X[:, 0].max())
print ('X[:, 0].min()', X[:, 0].min())
#Next, the following code will help us to visualize the dataset − 
plt.scatter(X[:, 0], X[:, 1], s = 20); # s effects the dot size
plt.show()

plt.scatter(X[:, 0], X[:, 1], c = y_true , s = 20, cmap = 'summer') # try with winter , spring , autumn, s affect the dot size
plt.show() 
 
####################### 3 d scatter
clusters = [[1,1,1],[5,5,5],[3,10,10]]
X, _ = make_blobs(n_samples = 150, centers = clusters, cluster_std = 0.60) #cluster_std deviation from the cluster
#After training the model, we store the coordinates for the cluster centers.
print ('--------------------' )
print (_)
#Finally, we plot the data points and centroids in a 3D graph.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], marker='o')