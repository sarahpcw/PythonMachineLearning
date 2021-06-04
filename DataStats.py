# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 01:44:56 2019
Getting statistics from data
Shape
Describe
Correlations
Skew
@author: u
"""

#Statistical Summary of Data
#We have discussed Python recipe to get the shape i.e. number of rows and columns, of data but many times we need to review the summaries out of that shape of data. It can be done with the help of describe() function of Pandas DataFrame that further provide the following 8 statistical properties of each & every data attribute −
#Count
#Mean
#Standard Deviation
#Minimum Value
#Maximum value
#25%
#Median i.e. 50%
#75%
#Example
from pandas import read_csv
from pandas import set_option
path = r"diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=names)
set_option('display.width', 100)
set_option('precision', 2)
print('Data Shape',data.shape)
print('Data Describe',data.describe())

#Reviewing Class Distribution
#Class distribution statistics is useful in classification problems where we need to 
#know the balance of class values. It is important to know class value distribution because if 
#we have highly imbalanced class distribution i.e. one class is having lots more observations than 
#other class, then it may need special handling at data preparation stage of our ML project. 
#We can easily get class distribution in Python with the help of Pandas DataFrame.
#Example
from pandas import read_csv
path = r"diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=names)
count_class = data.groupby('class').size()
print('Count Class and Group by',count_class)


#Reviewing Correlation between Attributes
#The relationship between two variables is called correlation. In statistics, the most common 
#method for calculating correlation is Pearson’s Correlation Coefficient. It can have three values 
#as follows −
##Coefficient value = 1 − It represents full positive correlation between variables.
##Coefficient value = -1 − It represents full negative correlation between variables.
##Coefficient value = 0 − It represents no correlation at all between variables.
##It is always good for us to review the pairwise correlations of the attributes in our dataset 
#before using it into ML project because some machine learning algorithms such as linear regression 
#and logistic regression will perform poorly if we have highly correlated attributes. In Python, we 
#can easily calculate a correlation matrix of dataset attributes with the help of corr() function on 
#Pandas DataFrame.
#Example
from pandas import read_csv
from pandas import set_option
path = r"diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=names)
set_option('display.width', 100)
set_option('precision', 2)
correlations = data.corr(method='pearson')
print('Correlations',correlations)
#
#Reviewing Skew of Attribute Distribution
##Skewness may be defined as the distribution that is assumed to be Gaussian but appears distorted 
#or shifted in one direction or another, or either to the left or right. 
#Reviewing the skewness of attributes is one of the important tasks due to following reasons −
##Presence of skewness in data requires the correction at data preparation stage so that we can get 
#more accuracy from our model.
##Most of the ML algorithms assumes that data has a Gaussian distribution 
#i.e. either normal of bell 
#curved data.
##In Python, we can easily calculate the skew of each attribute by using skew() function on Pandas 
#DataFrame.
#Example
from pandas import read_csv
path = r"diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=names)
print('SKEW',data.skew())