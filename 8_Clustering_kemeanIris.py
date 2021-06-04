# -*- coding: utf-8 -*-
"""
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
 
"""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pandas import read_csv
import numpy as np

################## example 0
X = np.array([[1, 2], [3, 4], [2, 0],    [10, 2], [8, 4], [7, 0]])
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)
lables = kmeans.labels_             #array([1, 1, 1, 0, 0, 0], dtype=int32)
kmeans.predict([[0, 0], [12, 3]])   #array([1, 0], dtype=int32)
centroids = kmeans.cluster_centers_
plt.scatter(X[:,0],X[:,1 ],   linewidths = 1 )
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
 
################## example 2
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
       }
df = DataFrame(Data,columns=['x','y'])
kmeans = KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()

################## example 3
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
       }
df = DataFrame(Data,columns=['x','y'])
  
kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()

################## example 4
path = r"C:\\Users\\u\\.spyder-py3\\MachineLearning\\A_Data\\iris.csv"
dataIris = read_csv(path)  #XIris = dataIris[:,1,3] 
XIris= dataIris.iloc[1:,1:2]   # row 1 all columns, col 1
clf = KMeans(n_clusters=6).fit(XIris)
centroids = clf.cluster_centers_   #print (centroids[:3, 0])    #print (centroids[3:6, 0])
#What is Cluster_centers_? #Predict the closest cluster each sample in X belongs to
labels = clf.labels_
plt.scatter(XIris.iloc[0:60,0],XIris.iloc[60:120, 0],linewidths = 1 )   # plot the data
plt.scatter(centroids[0:3, 0],centroids[3:6, 0], marker = "*",   linewidths = 1 ) # plot centroids into the dots

colors = ["g.","r.","c.","y."]
print ("\n--- centroids ---\n", centroids )
print (len( XIris.iloc[0:,0]) ) 
print (clf.labels_) 
