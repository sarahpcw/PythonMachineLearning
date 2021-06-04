# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 21:17:42 2019

https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
 
"""

import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
#Now, we need to load the input data −
iris = datasets.load_iris()
#From this dataset, we are taking first two features as follows −
X = iris.data[:, :2]
y = iris.target
#Next, we will plot the SVM boundaries with original data as follows −
x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#numpy.meshgrid(*xi, **kwargs)  #[source] 
#Return coordinate matrices from coordinate vectors.
#X_plot = np.c_[xx.ravel(), yy.ravel()] 
#numpy.c_ = Translates slice objects to concatenation along the second axis.
#numpy.ravel(a, order='C') #[source] #Return a contiguous flattened array.

########################################## 
"""
svm.svc kernel = LINEAR
"""

#Now, we need to provide the value of regularization parameter as follows −

#Next, SVM classifier object can be created as follows −
#( Kernel is the algorithm : kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
#Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. 
#If none is given, ‘rbf’ will be used.
Svc_classifier = svm.SVC(kernel='linear').fit(X, y)
#Linear Support Vector Classification.
##Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than 
#libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale 
#better to large numbers of samples.
#This class supports both dense and sparse input and the multiclass support is handled according to a 
#one-vs-the-rest scheme.
X_plot = np.c_[xx.ravel(), yy.ravel()]  #get the co-ordinates for the grid
Z = Svc_classifier.predict(X_plot) # Z is the prediction
print ( '----Score Linear', Svc_classifier.score ( X, y ) ) 

Z = Z.reshape(xx.shape)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
#contour and contourf draw contour lines and filled contours, respectively. 
#(contour is an outline representing or bounding the shape or form of something.)
#contourf(X, Y, Z,colors=('r', 'g', 'b'), # learn about the colours : https://chrisalbon.com/python/basics/set_the_color_of_a_matplotlib/
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Classifier with linear kernel')
#Output
#Text(0.5, 1.0, 'Support Vector Classifier with linear kernel')
# 
"""
"""
###########################################



########################################## 
"""
svm.svc kernel = rbf
"""
#For creating SVM classifier with rbf kernel, we can change the kernel to rbf as follows −
#Svc_classifier = svm.SVC(kernel = 'rbf', gamma ='auto',C = 1).fit(X, y)
Svc_classifier = svm.SVC(kernel = 'rbf', gamma ='auto').fit(X, y)
print ( '----Score RBF', Svc_classifier.score ( X, y ) ) 


Z = Svc_classifier.predict(X_plot)
print (Z)
Z = Z.reshape(xx.shape) 
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap = plt.cm.tab10, alpha = 0.2) # Z is the prediction in shades, alpha is the transparency
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1)  # SCatter the X 0 column and the X 1 Col
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Classifier with rbf kernel')

"""
"""
############################################