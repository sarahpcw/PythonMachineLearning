# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 23:43:35 2019
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
#https://towardsdatascience.com/machine-learning-algorithms-part-13-mean-shift-clustering-example-in-python-4d6452720b00
"""


###################### example 1
 
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
 
#We generate our own data using the make_blobs method.
clusters = [[1,1,1],[5,5,5],[3,10,10]]
X, _ = make_blobs(n_samples = 150, centers = clusters, cluster_std = 0.60)
#After training the model, we store the coordinates for the cluster centers.
ms = MeanShift()
ms.fit(X)
cluster_centers = ms.cluster_centers_
#Finally, we plot the data points and centroids in a 3D graph.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], marker='o')
ax.scatter(cluster_centers[:,0], cluster_centers[:,1], cluster_centers[:,2], marker='x', color='red', s=300, linewidth=5, zorder=10)
plt.show()