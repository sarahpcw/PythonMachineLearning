'''
Opening the diabetes with datasets.load and with a csv file
'''
import numpy as np
import pandas as pd
from sklearn import datasets 
diabetes = datasets.load_diabetes() # Load the diabetes dataset

print (diabetes.keys()) #dict_keys(['data', 'target', 'DESCR', 'feature_names'])
print (type ( diabetes['feature_names']) ) #<class 'list'>
print ('Feature names: ',  diabetes['feature_names'] ) #<class 'list'>
X_Data= diabetes['data']  # array
print ('len(X[:,0:])', len(X_Data[:,0:]))
y= diabetes['target']

indexL=list ( range(442) ) 
print ('len(indexL)',len(indexL))
 
df = pd.DataFrame(X_Data,index=indexL,columns=diabetes['feature_names']) #  pd.DataFrame( numbers or data,  rowindexes , columns names)
print (df)



from pandas import read_csv
path = r"diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#data = read_csv(path, names=headernames)
data = read_csv(path)  # comes in as pandas dataframe
print(data.head(0))
print(data.dtypes)
dataCSV = np.array(data)
print ( dataCSV)