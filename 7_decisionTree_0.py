# -*- coding: utf-8 -*-
"""
Classifier
After the training phase, a classifier can make a prediction.
Given a new feature vector, is the image an apple or an orange?
There are different types of classification algorithms, one of them is a decision tree.
If you have new data, the algorithm can decide which class you new data belongs.
The output will be [0] for apple and [1] for orange.
So this is new data and then we simply make the algorithm predicts.
"""
from sklearn import tree

features = [[0,50],[0,60],[1,35],[1,36],[1,40]]
labels = [0,0,1,1,1]

algorithm = tree.DecisionTreeClassifier()
algorithm = algorithm.fit(features, labels)

newData = [[0,51]]
print('prediction the classification for [0,51] will be : ')
print(algorithm.predict(newData))
