# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 13:58:07 2021

@author: u
"""
###################### diabetes
from sklearn import datasets
#Getting to know your data  diabetes
# Load the diabetes dataset
diabetes = datasets.load_diabetes()
print ('type', type (diabetes))
 
# Make sure the embedding works by default.
#data = datasets.load_diabetes()
print (diabetes.keys()) #dict_keys(['data', 'target', 'DESCR', 'feature_names'])
print (type ( diabetes['feature_names']) )
X, y = diabetes['data'], diabetes['target']
print (X.shape) # (442, 10)
print (y.shape) # (422,)

diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target
print (X)
print (y)
for key,val in diabetes.items():
    print (key, "=>", val)

#print (diabetes[:, 0:1] ) #cannot slice

###################### iris
from sklearn import datasets
iris = datasets.load_iris() # importing the dataset
iris.data # showing the iris data
#Here we import our necessary libraries. And import the iris dataset. And we print the data.
X=iris.data #assign the data to the X
y=iris.target #assign the target/flower type to the y

print (X.shape)#Then we print the size/shape of the variable X and y.
print (y.shape)