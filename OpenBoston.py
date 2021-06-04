# -*- coding: utf-8 -*-
"""
load_boston()          Load and return the boston house-prices dataset (regression).
load_iris()            Load and return the iris dataset (classification).
load_diabetes()        Load and return the diabetes dataset (regression).
load_digits([n_class]) Load and return the digits dataset (classification).
load_linnerud()        Load and return the linnerud dataset (multivariate regression).

THis file explores the iris data set
@"""

print (__doc__) # prints the comments above

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
boston = datasets.load_boston()
print ('Type:', type (boston))
 
    # Make sure the embedding works by default.
data = datasets.load_boston()
print ('Keys', data.keys()) #dict_keys(['data', 'target', 'DESCR', 'feature_names'])

print ('Desc:', data['DESCR'] )

print ('Type:',type ( data['feature_names']) )
print ('Feature names:', data['feature_names'] )
#Feature names: ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
fn = list (data['feature_names'] )
print (fn)

X, y = data['data'], data['target']
print (X.shape) # (442, 10)
print (y.shape) # (422,)
print (X)
print (y)