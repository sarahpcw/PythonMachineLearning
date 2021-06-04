"""
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
"""

from sklearn import datasets
iris = datasets.load_iris() #importing the dataset
iris.data       #showing the iris data

X=iris.data     #assign the data to the X
y=iris.target   #assign the target/flower type to the y
print (X.shape) #Then we print the size/shape of the variable X and y.
print (y.shape)

################## Split in test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=9) #Split the dataset

#################  Fit, predict and score
from sklearn.naive_bayes import GaussianNB
nv = GaussianNB() # create a classifier

nv.fit(X_train,y_train) # fitting the data 

from sklearn.metrics import accuracy_score

y_pred = nv.predict(X_test) # store the prediction data - it predict if the new x value will give me a  o,1,2 flower type
print ('Predicted y values for the list of X_test values ', y_pred)
print ('Accuracy score ' , accuracy_score(y_test,y_pred) ) # calculate the accuracy

#################  Visualise with Cnfusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_test)
print ( '---- confusion matrix ----- ')
print (cm)

#################  Standard report
from sklearn.metrics import classification_report
print ( '---- classification report ----')
print(classification_report(y_pred,y_test))

