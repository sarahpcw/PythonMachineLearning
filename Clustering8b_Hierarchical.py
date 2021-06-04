# -*- coding: utf-8 -*-
"""
matplotlib.pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
[source]

zip([iterable, ...])  This function returns a list of tuples

Tune the subplot layout.

"""

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
#Next, we will be plotting the datapoints we have taken for this example −

#################  hierarchical clustering with a dendogram
#From the above diagram, it is very easy to see that we have two clusters in out datapoints but in the real world data, there can be thousands of clusters. Next, we will be plotting the dendrograms of our datapoints by using Scipy library −
from scipy.cluster.hierarchy import dendrogram, linkage


X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]

linked = linkage(X, 'single')

labelList = range(1, 11)

plt.figure(figsize = (5,3))

dendrogram(linked, orientation = 'top',labels = labelList, distance_sort ='descending', show_leaf_counts = True)

plt.show()









################### agglomorative clustering
##Now, once the big cluster is formed, the longest vertical distance is selected.'
# A vertical line is then drawn through it as shown in the following diagram. As the horizontal 
# line crosses the blue line at two points, the number of clusters would be two.
X = np.array(
   [[7,8],[12,20],[17,19],[26,15],[32,37],[87,75],[73,85], [62,80],[73,60],[87,96],])
labels = range(1, 11)
#plt.figure(figsize = (10, 7))
#plt.subplots(2,1,1)
plt.scatter(X[:,0],X[:,1],  label = 'True Position')
for label, x, y in zip(labels, X[:, 0], X[:, 1]):
   plt.annotate(
      label,xy = (x, y), xytext = (-3, 3),textcoords = 'offset points', ha = 'right', va = 'bottom')
plt.show()
#Next, we need to import the class for clustering and call its fit_predict method to predict the cluster. We are importing AgglomerativeClustering class of sklearn.cluster library −
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
cluster.fit(X)
#print  (cluster.fit_predict(X))
X = np.array(
   [[7,8],[12,20],[17,19],[26,15],[32,37],[87,75],[73,85], [62,80],[73,60],[50,50],])
cluster.fit_predict(X)
#Next, plot the cluster with the help of following code −
#plt.subplots(2,1,2)
plt.scatter(X[:,0],X[:,1], c = cluster.labels_, cmap = 'rainbow')
 
#The above diagram shows the two clusters from our datapoints.

###################################################################################
