# -*- coding: utf-8 -*-
"""
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
"""
import numpy as np
import pylab as pl
from sklearn import neighbors, datasets

################################## import data
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features. 
Y = iris.target  #flower types

################################## Split in train/test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=9) 

################################## Fit, predict and score
knn=neighbors.KNeighborsClassifier()
# we create an instance of Neighbours Classifier and fit the data.
knn.fit(X, Y)

y_pred = knn.predict( X_test )
print ('y_pred',y_pred)

accuracy = knn.score(X_test, y_test) 
print ( 'accuracy 1:',  accuracy )
accuracy = knn.score(X_test, y_pred) 
print ( 'accuracy 1:',  accuracy )


################################## Visualise
x_min =  X[:,0].min() - .5
x_max =  X[:,0].max() + .5
y_min =  X[:,1].min() - .5
y_max =  X[:,1].max() + .5
h = .05 # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) #Return coordinate matrices from coordinate vectors.
print ( (xx.shape)) #(68, 92)
print ( yy.shape) # also (68, 92)
Z = knn.predict( np.c_[xx.ravel(), yy.ravel()] )   
#
print ( 'Z', Z , Z.shape)  # z.shape also (68, 92)
Z = Z.reshape(xx.shape)
# Put the result into a color plot
pl.figure(1, figsize=(5,4))
#pl.set_cmap(pl.cm.Paired)  #set_cmap(self, cmap), [source] set the colormap for luminance data
pl.pcolormesh(xx, yy,Z) # places the Z ( predicted values ) in the background

pl.scatter(X[:,0], X[:,1]  ) # all rows of col 0 , vs all rows of col 1
pl.xlabel('Sepal length')
pl.ylabel('Sepal width')

pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())

pl.show()