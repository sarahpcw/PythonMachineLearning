# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 21:01:46 2019

#numpy.c_¶
#numpy.c_ = <numpy.lib.index_tricks.CClass object>
#Translates slice objects to concatenation along the second axis.
#This is short-hand for np.r_['-1,2,0', index expression], which is useful because of its common occurrence. In particular, arrays will be stacked along their last axis after being upgraded to at least 2-D with 1’s post-pended to the shape (column vectors made out of 1-D arrays).
#See also 
#column_stack
#Stack 1-D arrays as columns into a 2-D array.
#r_
#For more detailed documentation.
#Examples
@author: u
""" 

import numpy as np

n = np.c_[np.array([1,2,3]), np.array([4,5,6])]
print  (n )
#array([[1, 4],
#       [2, 5],
#       [3, 6]])
n = np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
#array([[1, 2, 3, ..., 4, 5, 6]])
print (n)