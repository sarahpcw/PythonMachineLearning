# -*- coding: utf-8 -*-
"""
Numpy polyfit() method is used to fit our data inside a polynomial function.
The numpy.poly1d() function helps to define a polynomial function. 
It makes it easy to apply “natural operations” on polynomials.
#https://www.geeksforgeeks.org/numpy-poly1d-in-python/
"""

import numpy as np
import matplotlib.pyplot as plt

##################### Get the data
X = [1, 5, 8, 10, 14, 18]
Y = [1, 1, 10, 20, 45, 75]
# Train Algorithm (Polynomial)
degree = 2

##################### Fit, predict and score
poly_fit = np.poly1d(np.polyfit(X,Y, degree)) #trains the data
# Predict price
print( poly_fit(12) )

##################### Visualise
# Plot data
xx = np.linspace(0, 26, 100) #predit for xx (gets 100 points between 0 am=nd 26, take these 100 numbers and fit to polyfit)

plt.plot(xx, poly_fit(xx), c='r',linestyle='--', color='orange') #this draws the line
plt.title('Polynomial')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis([0, 25, 0, 100]) # values on the axis
plt.grid(True)
plt.scatter(X, Y) #scatter x and why on the line
plt.show()


