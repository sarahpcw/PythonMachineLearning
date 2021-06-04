# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:33:14 2019

@author: u
"""

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
#Next, we will load the diabetes dataset and create its object −
diabetes = datasets.load_diabetes()
#As we are implementing SLR, we will be using only one feature as follows −
X = diabetes.data[:, np.newaxis, 2]
#Next, we need to split the data into training and testing sets as follows −
X_train = X[:-30]
X_test = X[-30:]
#Next, we need to split the target into training and testing sets as follows −
y_train = diabetes.target[:-30]
y_test = diabetes.target[-30:]
#Now, to train the model we need to create linear regression object as follows −
regr = linear_model.LinearRegression()
#Next, train the model using the training sets as follows −
regr.fit(X_train, y_train)
#Next, make predictions using the testing set as follows −
y_pred = regr.predict(X_test)
#Next, we will be printing some coefficient like MSE, Variance score etc. as follows −
print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))
#Now, plot the outputs as follows −
plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_test, y_pred, color = 'red', linewidth = 3)
plt.xticks(())
plt.yticks(())
plt.show()
#Output
#Coefficients:
#   [941.43097333]
#Mean squared error: 3035.06
#Variance score: 0.41