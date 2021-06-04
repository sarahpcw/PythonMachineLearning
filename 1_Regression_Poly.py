# -*- coding: utf-8 -*-
"""
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
##################### Get the data
dataset = pd.read_csv('https://s3.us-west-2.amazonaws.com/public.gamelab.fun/dataset/position_salaries.csv')
print (dataset.keys())
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
print ( dataset.iloc[:,:])
print ('Keys',dataset.iloc[:,0:1])
print (dataset)
print ('type', type (dataset)) #type <class 'pandas.core.frame.DataFrame'>
    # Make sure the embedding works by default.
print ('Keys',dataset.keys())  #Keys Index(['Position', 'Level', 'Salary'], dtype='object')
print (X.shape) # (10, 1)
print (y.shape) # (10,)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

############### Example 1:
# Visualizing the Linear Regression results
def viz_linear():
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg.predict(X), color='green')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return

################## Fit, predict and score: 
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)  # fit to training data
# Predicting a new result with Linear Regression
print ('Lin reg predict 5.5',lin_reg.predict([[5.5]]))
print ('Lin reg predict 8.0',lin_reg.predict([[8.0]]))
print ('Lin reg predict 9.7',lin_reg.predict([[9.7]]))

################# Visualise
viz_linear()





############### Example 2:
def viz_polymonial_smooth():
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape(len(X_grid), 1) 
    #Why do we need to reshape? (https://www.tutorialspoint.com/numpy/numpy_reshape.htm)
    # Visualizing the Polymonial Regression results
    plt.scatter(X, y, color='red')
    plt.plot(X_grid, pol_reg.predict(poly_reg.fit_transform(X_grid)), color='blue')
    plt.title('Truth or Bluff (Poly Smooth Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return

# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='pink')
    plt.title('Truth or Bluff (Poly Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return


############### Fit, predict and score:
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
# Fit
X_poly   = poly_reg.fit_transform(X)
pol_reg  = LinearRegression()
pol_reg.fit(X_poly, y)

# Predicting a new result with Polymonial Regression
print ( pol_reg.predict(poly_reg.fit_transform([[5.5]])) )
print ( pol_reg.predict(poly_reg.fit_transform([[8.0]])) )
print ( pol_reg.predict(poly_reg.fit_transform([[9.7]])) )

############### Visualise the result
viz_polymonial()

# Additional feature, Making the plot line (Blue one) more smooth
viz_polymonial_smooth()

############### Save the plot
fig = plt.figure()
fig.savefig('C:/Users/u/testPlot.pdf')
fig.savefig('C:/Users/u/testPlot.png')
fig.savefig('C:/Users/u/testPlot.jpg')

