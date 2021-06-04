# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 18:21:04 2021

@author: u
"""


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

################## Get the data
X = [[4], [8], [12], [16], [18]]
y = [[40000], [80000], [100000], [120000], [150000]]


################## Fit, predict and score: 
model = LinearRegression()
model.fit(X,y)

# predict
rooms = 11
prediction = model.predict([[rooms]])
print('Price prediction: $%.2f' % prediction)


#################### Visualise
# Plot outputs
plt.scatter(X, y,  color='black' )
plt.plot(   X, y,  color='blue', linewidth=3  )

plt.xticks(()) # challenge: Research MatPlotLib to plot the ticks on the axes
plt.yticks(())

plt.show()