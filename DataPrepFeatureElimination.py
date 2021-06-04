# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 23:03:56 2019

@author: u
"""
#
#Recursive Feature Elimination
#As the name suggests, RFE (Recursive feature elimination) feature selection technique removes the attributes 
#recursively and builds the model with remaining attributes. We can implement RFE feature selection technique with the help of RFE class of scikit-learn Python library.
#Example
#In this example, we will use RFE with logistic regression algorithm to select the best 3 attributes having the 
#best features from Pima Indians Diabetes dataset to.

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error #, r2_sco
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

import csv
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from pandas import read_csv

path = r"iris.csv"
#Next, use the csv.reader()function to read data from CSV file −
with open(path,'r') as f:
    reader = csv.reader(f,delimiter = ',')
    headers = next(reader)
    data = list(reader)
#    print (data)

print(headers)
#The following line of script will print the shape of the data i.e. number of rows & columns in the file −

#Next script line will give the first three line of data file −
print(data[0:3])
print ( data)