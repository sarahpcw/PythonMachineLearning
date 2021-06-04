# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 00:35:37 2019
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

############### Split is test/train
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples = 400, centers = 6, cluster_std = 0.60, random_state = 1) #cluster_std :The standard deviation of the clusters.

######### Fit, predict and score: choose the model with desired number of clusters, fit, predict, get the centres and labels 
kmeans = KMeans(n_clusters = 6)
kmeans.fit(X)
y_predict = kmeans.predict(X)
centers = kmeans.cluster_centers_

print ( kmeans.labels_)
print ( kmeans.score ( X, y_predict ) )

print (' len ( y_true )' ,len ( y_true ) ) 
print ( 'len (y_kmeans )',len (y_predict  ) )

########## Visualise
#Next, the following code will help us to visualize the dataset − 
plt.scatter(X[:, 0], X[:, 1], s = 20);
plt.show()
 
plt.scatter(X[:, 0], X[:, 1], c = y_predict , s = 20, cmap = 'summer')
plt.scatter(centers[:, 0], centers[:, 1], c = 'blue', s = 100, alpha = 0.9) ####Scatter the clusters Coordinates of cluster centers. I
plt.show() 
