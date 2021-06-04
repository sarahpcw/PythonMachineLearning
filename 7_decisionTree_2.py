# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 21:34:34 2019
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Define training and target set for the classifier
train = [[1,2,3]
,[2,5,1]
,[2,1,7]]
target = [10,20,30]

# Initialize Classifier. 
# Random values are initialized with always the same random seed of value 0 
# (allows reproducible results)
dectree = DecisionTreeClassifier(random_state=0)
dectree.fit(train, target)

# Test classifier with other, unknown feature vector
 
test = [5,5,5]
test = np.asarray(test) #convert the list to an array, reshape to get it into a 2d and then predict
test= test.reshape(1, -1)
predicted = dectree.predict(test) 

print ('predicted',predicted)
#Output can be visualized using:
print ('score', dectree.score(test, predicted))
 
