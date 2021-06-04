# -*- coding: utf-8 -*-
"""
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
#Step 3: Build a dataframe
#For this step, you’ll need to capture the dataset (from step 1) in Python. 
#You can accomplish this task using pandas Dataframe:

############################ DATA
candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'admitted': [1,1,1,1,1,1,0,1,1,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }

df = pd.DataFrame(candidates,columns= ['gmat', 'gpa','work_experience','admitted'])
print (df)
#Alternatively, you could import the data into Python from an external file.
#Step 4: Create the logistic regression in Python
#Now, set the independent variables (represented as X) and the dependent variable (represented as y) :
X = df[['gmat', 'gpa','work_experience']]
y = df['admitted']

############################ Split is test and train, fit, predict and score
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

############################ Fit, predict, score
logistic_regression= LogisticRegression() #Apply the logistic regression \

logistic_regression.fit(X_train,y_train)

y_pred=logistic_regression.predict(X_test) 
 
print('-----------Accuracy:------------- ',metrics.accuracy_score(y_test, y_pred))

############################ Visualise on a confusion matric
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
print ('--------confusion matrix ------\n')
print (confusion_matrix)
