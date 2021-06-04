# -*- coding: utf-8 -*-
"""
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

https://www.youtube.com/watch?v=CPqOCI0ahss&t=228s

"""


from sklearn import datasets
iris = datasets.load_iris() # importing the dataset
iris.data # showing the iris data
#Explain:
#Here we import our necessary libraries. And import the iris dataset. And we print the data.
X=iris.data #assign the data to the X
y=iris.target #assign the target/flower type to the y

print (X.shape)#Then we print the size/shape of the variable X and y.
print (y.shape)
print(iris.target_names)
# print the names of the four features
print(iris.feature_names)


#Here we split our data set into train and test as X_train, X_test, y_train, and y_test.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2) #Split the dataset (use random state as well to fix results)


# supervised classification
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()  
classifier.fit(X_train, y_train)    
y_pred = classifier.predict(X_test)


print ('Prediction naive bays: ---------')
print ( y_pred )
from sklearn import metrics
print("Accuracy  naive bays:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print ( '---- confusion matrix  naive bays:----- ')
print (cm)
import seaborn as sn
sn.heatmap(cm, annot=True)
print ('--------confusion matrix ------\n')
print (cm)