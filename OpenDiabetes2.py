# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 07:05:42 2019

@author: u

"""
#Load CSV with Python Standard Library
import csv
#Next, we need to import Numpy module for converting the loaded data into NumPy array.
import numpy as np
#Now, provide the full path of the file, stored on our local directory, having the CSV data file −
path = r"iris.csv"
#Next, use the csv.reader()function to read data from CSV file −
with open(path,'r') as f:
    reader = csv.reader(f,delimiter = ',')
    headers = next(reader)
    data = list(reader)
#    data = np.array(data).astype(float)
#We can print the names of the headers with the following line of script −
print(headers)
#The following line of script will print the shape of the data i.e. number of rows & columns in the file −
#print(data.shape)
#Next script line will give the first three line of data file −
print(data[:3])
count = 0
for line in data:
    count+=1
    if count < 10:
        print (line)

#Load CSV with NumPy
#Another approach to load CSV data file is NumPy and numpy.loadtxt() function. 
#The following is an example of loading CSV data file with the help of it −
from numpy import loadtxt
print ( 'loadtext -- does not like text in the data')
#path = r"C:\pima-indians-diabetes.csv"
path=r"loadtexttestdata.csv"
datapath= open(path, 'r')
data = loadtxt(datapath, delimiter=",")
print(data.shape)
print(data[:3])


#Load CSV with Pandas
#Another approach to load CSV data file is by Pandas and pandas.read_csv()function. 
#This is the very flexible function that returns 
#a pandas.DataFrame which can be used immediately for plotting. 
#The following is an example of loading CSV data file with the help of it −
from pandas import read_csv
#path = r"C:\pima-indians-diabetes.csv"
path=r"diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
print(data.shape)
print(data[:3])
print(data.head(50))


from pandas import read_csv
path = r"iris.csv"
data = read_csv(path)
print(data.shape)
print(data[:3])
print ('SKEW:')
print(data.skew())
