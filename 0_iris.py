# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 08:09:47 2019
@author: u
"""
from sklearn import datasets
#Load dataset
iris = datasets.load_iris()
print(iris.target_names)# print the label species(setosa, versicolor,virginica)
print(iris.feature_names)#['setosa' 'versicolor' 'virginica']#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris.data[0:5])# print the iris data (top 5 records)
print(iris.target)# print the iris labels (0:setosa, 1:versicolor, 2:virginica)
 
#Load dataset
iris = datasets.load_iris()
for key,val in iris.items():
    print ( val)
    
import pandas as pd
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
data.head()

X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  # Labels
for key,val in data.items():
    print (key, "=>", val)