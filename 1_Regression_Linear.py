"""
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html 
#from sklearn.model_selection import train_test_split 
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

##################### Get the data
# Load the diabetes dataset
diabetes = datasets.load_diabetes()
print ('1 type (diabetes)', type (diabetes))
print ('2 diabetes.keys()', diabetes.keys()) #dict_keys(['data', 'target', 'DESCR', 'feature_names'])

print ("3 type ( diabetes['feature_names'])" ,type ( diabetes['feature_names']) )

X, y = diabetes['data'], diabetes['target']
print ('4 X.shape' , X.shape) # (442, 10)  # this is a numpy array
print ('5 y.shape' , y.shape) # (422,)
#print (diabetes.shape )


##################### Split data in train and test
# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
#print ('diabetes_X_test.shape',diabetes_X_test)
# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]


##################### Fit, predict and score
# Fit : Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train) # Train the model using the training sets

# Predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)
print('diabetes_y_pred: ', diabetes_y_pred)

print('Coefficients: ', regr.coef_) # The coefficients

# Score
from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred)) # sum of r squared


 
##################### Visualise the data
# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(()) # challenge: Research MatPlotLib to plot the ticks on the axes
plt.yticks(())

plt.show()
