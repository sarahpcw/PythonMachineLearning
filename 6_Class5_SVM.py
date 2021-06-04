# -*- coding: utf-8 -*-
""" 
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
########################## Data
iris = datasets.load_iris()
#From this dataset, we are taking first two features as follows −
X = iris.data[:, :2]
y = iris.target

Y_score = np.c_[y.ravel()]
#Next, we will plot the SVM boundaries with original data as follows −
x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()] 
#numpy.c_ = Translates slice objects to concatenation along the second axis.


########################## Fit and predict
Svc_classifier = svm.SVC(kernel='linear').fit(X, y)
Z = Svc_classifier.predict(X_plot)

########################## Visualise
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3) # prints the background
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1) # prints the dots on top of the background
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Classifier with linear kernel')
 

########################## Fit and predict
Svc_classifier = svm.SVC(kernel = 'rbf', gamma ='auto').fit(X, y)
Z = Svc_classifier.predict(X_plot)
########################## Visualise
Z = Z.reshape(xx.shape) 
plt.figure(figsize=(15, 5))
plt.subplot(121) # can be omitted , then it is just a little wider
plt.contourf(xx, yy, Z, cmap = plt.cm.tab10, alpha = 0.3) # Z is the prediction in shades
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1)  # SCatter the X 0 column and the X 1 Col
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Classifier with rbf kernel')