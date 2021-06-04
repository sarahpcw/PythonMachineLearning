# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:44:29 2019

@author: u
"""
import numpy as np
###############Ravel
#if all you want is calling ravel on your (nested, I s'pose?) list, you can do that directly, 
#numpy will do the casting for you:
#


L = [[1,None,3],["The", "quick", object]]
print(L)
np.ravel(L)
print(np.ravel(L)) # array([1, None, 3, 'The', 'quick', <class 'object'>], dtype=object)

L = [[1,2,3],["The", "quick", "fox"]]
print(L)
np.ravel(L)
print(np.ravel(L)) 


L = [[1,2,3],["The", "quick", "fox"], [5,6,7]]
print(L)
np.ravel(L)
print(np.ravel(L)) 

## c_
z = np.ravel(L)
X_plot = np.c_[z] 
print (X_plot)
#numpy.c_ = Translates slice objects to concatenation along the second axis.
#numpy.ravel(a, order='C') #[source] #Return a contiguous flattened array.