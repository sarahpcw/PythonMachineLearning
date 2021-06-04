import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
print ('1 type (diabetes)', type (diabetes))
print ('2 diabetes.keys()', diabetes.keys()) #dict_keys(['data', 'target', 'DESCR', 'feature_names'])
print ("3 type ( diabetes['feature_names'])" ,type ( diabetes['feature_names']) )
print ("4 data ( diabetes['data'])" ,type ( diabetes['data']) )
print (diabetes.data[:])
print (diabetes.data)
print (diabetes.target[:])
print (diabetes.DESCR[:])
print (diabetes.feature_names[:])

X, y = diabetes['data'], diabetes['target']
print ('4 X.shape' , X.shape) # (442, 10)  # this is a numpy array
print ('5 y.shape' , y.shape) # (422,)


#print (diabetes.shape )
# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

x = np.arange(100)  # 3 down 5 accross
print ( x[:-20])  # from 0 to 20 from the end
print ( x[-20:]) # from the begiiing to 20 less from the end , to the end

print ( x[:20])  # all but the last 20,  0 - 70
print ( x[20:]) # from 20 to the end  