# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:27:44 2018
Upsupervised
p 34 
Clustering

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans 
####################### Data
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

plt.scatter(X[:, 0],X[:, 1], s=150, linewidths = 5, zorder = 10)
plt.show()
 
###################### Fit, predict, score
clf = KMeans(n_clusters=2)
clf.fit(X)
 
centroids = clf.cluster_centers_
labels = clf.labels_

###################### Visualise
colors = ["g.","r.","c.","y."]
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)
plt.show()
print ( centroids) 
print ( labels )