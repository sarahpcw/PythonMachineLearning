# -*- coding: utf-8 -*-
"""
KNN
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
@author: pcworkshopslondon.co.uk 
"""
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
  
################### Data: loading the iris dataset 
iris = datasets.load_iris() 
# X -> features, y -> label 
X = iris.data 
y = iris.target 
  
################### Split:  dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 

################### Fit, predict and score
# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
  
knn_predictions = knn.predict(X_test)  

# accuracy on X_test 
accuracy = knn.score(X_test, y_test) 
print ( 'accuracy 1:',  accuracy )

################### Visualise with confudion matrix
knn_predictions = knn.predict(X_test)  
cm = confusion_matrix(y_test, knn_predictions) #oorspronklike y_test vs the "predicted y_test"
print (y_test)
print (cm)



"""
n=total nrs predicted no --- predicted yes --- total
actual no   tn =             fp =           ---   
actual yes  fn =             tp =           ---   
totals: 
                       
""" #there are 13 0's , 16 1's and 9 2's
"""
Predicted  0  1  2  All
True                   
0          13  0  0    13
1          0   15  1   16
2          0    0  9    9
All        13   15 10   38
"""