# -*- coding: utf-8 -*-
"""
https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html#scipy.cluster.hierarchy.dendrogram
"""
#Example 2
#As we understood the concept of dendrograms from the simple example discussed above,
#another example:   creating clusters of the data point in Pima Indian Diabetes Dataset by using hierarchical clustering.

import matplotlib.pyplot as plt
from pandas import read_csv

######################## Data
path = r"C:\\Users\\u\\.spyder-py3\\MachineLearning\\A_Data\\diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names = headernames)
array = data.values

####################### Split
X = array[:,0:8]
Y = array[:,8]
patient_data = data.iloc[1:, 3:5].values
print (data.shape)  #(768, 9)
print ( data.head())
print (patient_data  )

####################### Visualise with dendogram
import scipy.cluster.hierarchy as shc
plt.figure(figsize = (5, 4))
plt.title("Patient Dendograms")
dend = shc.dendrogram(shc.linkage(patient_data , method = 'ward')) #linkage Performs hierarchical/agglomerative clustering.

####################### Visualise with many dots  
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
cluster.fit_predict(patient_data)
plt.figure(figsize = (10, 7))
plt.scatter(patient_data[:,0], patient_data[:,1], c = cluster.labels_, cmap = 'rainbow')
#plt.scatter(patient_data[:,0], patient_data[:,1], c = cluster.labels_ )