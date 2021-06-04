# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:00:41 2019
1. remove headings
2. Scaling the data
3. Binarise the data
4. Standardise the data
5. Labelise the data
6. encode / decode the data

@author: u
"""

import numpy as np
from pandas import read_csv
from numpy import set_printoptions
from sklearn import preprocessing
path = r'diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
print ( dataframe.values)  # take off the heading
array = dataframe.values[0:4,0:4]

print ( dataframe.values[1:,:])  # take off the heading
array = dataframe.values[1:,:]
#Now, we can use MinMaxScaler class to rescale the data in the range of 0 and 1.
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_rescaled = data_scaler.fit_transform(array)
###We can also summarize the data for output as per our choice. Here, we are setting the precision to 1 and showing the first 10 rows in the output.
set_printoptions(precision=1)
print ("\nScaled data:\n", data_rescaled[0:10])


#binirazation
from pandas import read_csv
from sklearn.preprocessing import Binarizer
path = r'diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
print ( dataframe.values[1:,:])  # take off the heading
array = dataframe.values[1:,:]
#Now, we can use Binarize class to convert the data into binary values.
binarizer = Binarizer(threshold=0.5).fit(array)
Data_binarized = binarizer.transform(array)
#Here, we are showing the first 5 rows in the output.
print ("\nBinary data:\n", Data_binarized [0:5])




#standardisation
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from numpy import set_printoptions
path = r'diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values[1:,:]
#Now, we can use StandardScaler class to rescale the data.
data_scaler = StandardScaler().fit(array)
data_rescaled = data_scaler.transform(array)
#We can also summarize the data for output as per our choice. Here, we are setting the precision to 2 and showing the first 5 rows in the output.
set_printoptions(precision=2)
print ("\nRescaled data:\n", data_rescaled [0:5])

#Data Labeling
#We discussed the importance of good fata for ML algorithms as well as some techniques to pre-process the data before sending it to ML algorithms. One more aspect in this regard is data labeling. It is also very important to send the data to ML algorithms having proper labeling. For example, in case of classification problems, lot of labels in the form of words, numbers etc. are there on the data.
#What is Label Encoding?
#Most of the sklearn functions expect that the data with number labels rather than word labels. Hence, we need to convert such labels into number labels. This process is called label encoding. We can perform label encoding of data with the help of LabelEncoder() function of scikit-learn Python library.
#Example
#In the following example, Python script will perform the label encoding.
#First, import the required Python libraries as follows −
import numpy as np
from sklearn import preprocessing
#Now, we need to provide the input labels as follows −
input_labels = ['red','black','red','green','black','yellow','white']
#The next line of code will create the label encoder and train it.
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
#The next lines of script will check the performance by encoding the random ordered list −
test_labels = ['green','red','black']
encoded_values = encoder.transform(test_labels)
print("\n+++++++++++\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))
encoded_values = [3,0,4,1]
decoded_list = encoder.inverse_transform(encoded_values)
#We can get the list of encoded values with the help of following python script −
print("\n+++++++++++++++\nEncoded values =", encoded_values)
print("Decoded labels =", list(decoded_list))