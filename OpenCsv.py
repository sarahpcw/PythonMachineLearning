# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 21:21:29 2019

@author: u
"""
"""
File Header
In CSV data files, the header contains the information for each field. We must use the same delimiter for the header file and for data file because it is the header file that specifies how should data fields be interpreted.
The following are the two cases related to CSV file header which must be considered −
Case-I: When Data file is having a file header − It will automatically assign the names to each column of data if data file is having a file header.
Case-II: When Data file is not having a file header − We need to assign the names to each column of data manually if data file is not having a file header.
In both the cases, we must need to specify explicitly weather our CSV file contains header or not.
"""
import pandas as pd
df = pd.read_csv('C:\\Users\\u\\.spyder-py3\\DataD1\\MBPlayerSalaries200Sample2.csv')
print(df.head(2)) 
#Writng To A Csv File
df.sample(200).to_csv('MBPlayerSalaries200Sample22.csv')


from pandas import read_csv
path = r"C:\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv('C:\\Users\\u\\.spyder-py3\\DataD1\\MBPlayerSalaries200Sample2.csv', names=headernames)
print('Shape', data.shape)
print('data 3', data[:3])


count = 0
fileN = open('C:\\Users\\u\\.spyder-py3\\DataD1\\MBPlayerSalaries200Sample2.csv', "r")
for each in fileN:
    count += 1
    if count == 1:
        print (each)
#    if count < 10:
#        print ('row', each)
fileN.close()


import csv
#Next, we need to import Numpy module for converting the loaded data into NumPy array.
import numpy as np
#Now, provide the full path of the file, stored on our local directory, having the CSV data file −
path = 'C:\\Users\\u\\.spyder-py3\\DataD1\\MBPlayerSalaries200Sample2.csv'
#Next, use the csv.reader()function to read data from CSV file −
with open(path,'r') as f:
    reader = csv.reader(f,delimiter = ',')
    headers = next(reader)
    print ('headers 2: ', headers)
    data = list(reader)
#    print (data)
#    data = np.array(data).astype(float)
#    print(data.shape)
#    Next script line will give the first three line of data file −
    print('Data 3 again', data[:3])
    
