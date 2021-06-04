# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:47:54 2019
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
"""
from sklearn import tree

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

#After being fitted, the model can then be used to predict the class of samples:
clf.predict([[2., 2.]]) 
print ( clf.predict([[2., 2.]])  )  #array([1])
#Alternatively, the probability of each class can be predicted, which is the fraction of training samples of the same class in a leaf:
clf.predict_proba([[2., 2.]])  
print ( clf.predict_proba([[2., 2.]])  ) #array([[0., 1.]])

print ( "score" , clf.score([[2., 2.]], clf.predict([[2., 2.]]) ))
 
