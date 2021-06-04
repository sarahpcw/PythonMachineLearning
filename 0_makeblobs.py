# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 19:46:40 2019

@author: u
"""

import numpy as np
import matplotlib.pyplot as plt
# what is random state: https://scikit-learn.org/stable/glossary.html#term-random_state
#https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
#sklearn.datasets.make_blobs(n_samples=100, n_features=2, *, centers=None,
#                            cluster_std=1.0, center_box=- 10.0, 10.0, 
#                            shuffle=True, random_state=None, return_centers=False)

#Now, by using make_blobs() function of Scikit learn,
# we can generate blobs of points with Gaussian distribution 
# as follows −
from sklearn.datasets import make_blobs
X, y      = make_blobs(300, centers = 2, cluster_std = 1.5, random_state = 2,)  #make blobs return both x and y
plt.scatter(X[:, 0], X[:, 1], c = y, s = 20, cmap = 'winter');
#print ( 'X[:, 0] --------')
#print (  X[:, 0] )
#print ( 'X[:, 1] --------')
#print ( X[:, 1])
#print ( y ) # y is the target ( 0 and 1 's )



clusters  = [[1,1,1],[5,5,5],[3,10,10]]
X, _      = make_blobs(n_samples = 150, centers = clusters, cluster_std = 0.60)
X, y_true = make_blobs(n_samples = 400, centers = 6,        cluster_std = 0.60, random_state = 1) #cluster_std :The standard deviation of the clusters.


#rng = np.random.RandomState(0)
#Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)  # generating more blobs for x 
#print (type ( Xnew), Xnew.shape )
##
#
#from sklearn.naive_bayes import GaussianNB
#model_GBN = GaussianNB()
#model_GBN.fit(X, y);
##Now, we have to do prediction. It can be done after generating some new data as follows −
#ynew = model_GBN.predict(Xnew)
#print (ynew)