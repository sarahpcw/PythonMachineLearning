# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:05:08 2019

@author: u
"""
import numpy as np
a1 = np.arange(100) * 0.98
a1 = a1.reshape(10,10)
print ( a1[:,:])
a1=a1[:, np.newaxis, 2]
print (a1.shape)
print ('a1')
print (a1)


#Simply put, the newaxis is used to increase the dimension of the existing array by one more dimension
#, when used once. Thus,
#1D array will become 2D array
#2D array will become 3D array
#3D array will become 4D array
#4D array will become 5D array


#Scenario: np.newaxis might come in handy when you want to explicitly 
#convert a 1D array to either a row vector or a column vector, as depicted in the above picture.
#Example:
# 1D array
arr = np.arange(4)
print ( arr.shape )
#Out[8]: (4,)

# make it as row vector by inserting an axis along first dimension
row_vec = arr[np.newaxis, :]     # arr[None, :]
print ( row_vec.shape )
#Out[10]: (1, 4)

# make it as column vector by inserting an axis along second dimension
col_vec = arr[:, np.newaxis]     # arr[:, None]
print ( col_vec.shape ) 
#Out[12]: (4, 1)










print ('a1[:-2]')
diabetes_X_train = a1[:-2] # 2 columns, from the last backwards
print ( diabetes_X_train)
print ('a1[-2:]')
diabetes_X_test = a1[-2:]  # 2 rows from the last backwards
print ( diabetes_X_test  )

