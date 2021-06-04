# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 01:46:16 2019

@author: u
"""
#
#####################Univariate Selection
##This feature selection technique is very useful in selecting those features, with the help of 
#statistical testing, having strongest relationship with the prediction variables. 
#We can implement univariate feature selection technique with the help of SelectKBest0class of
# scikit-learn Python library.
#Example
#In this example, we will use Pima Indians Diabetes dataset to select 4 of the attributes having 
#best features with the help of chi-square statistical test.
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
path = r'diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values
#Next, we will separate array into input and output components −
X = array[:,0:8]
Y = array[:,8]
#The following lines of code will select the best features from dataset −
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X,Y)
#We can also summarize the data for output as per our choice. Here, we are setting the precision to 2 and showing the 4 data attributes with best features along with best score of each attribute −
set_printoptions(precision=2)
print(fit.scores_)
featured_data = fit.transform(X)
print ("\nFeatured data:\n", featured_data[0:4])
#Output
#[ 111.52 1411.89 17.61 53.11 2175.57 127.67 5.39 181.3 ]
#Featured data:
#[[148.  0. 33.6 50. ]
#[  85.  0. 26.6 31. ]
#[ 183.  0. 23.3 32. ]
#[  89. 94. 28.1 21. ]]


###################Recursive Feature Elimination
#As the name suggests, RFE (Recursive feature elimination) feature selection technique removes the attributes recursively and builds the model with remaining attributes. We can implement RFE feature selection technique with the help of RFE class of scikit-learn Python library.
#Example
#In this example, we will use RFE with logistic regression algorithm to select the best 3 attributes having the best features from Pima Indians Diabetes dataset to.
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
path = r'diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values
#Next, we will separate the array into its input and output components −
X = array[:,0:8]
Y = array[:,8]
#The following lines of code will select the best features from a dataset −
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Number of Features: %d")
print("Selected Features: %s")
print("Feature Ranking: %s")
#Output
#Number of Features: 3
#Selected Features: [ True False False False False True True False]
#Feature Ranking: [1 2 3 5 6 1 1 4]
#We can see in above output, RFE choose preg, mass and pedi as the first 3 best features. They are marked as 1 in the output.

###############Principal Component Analysis (PCA)
#PCA, generally called data reduction technique, is very useful feature selection technique as it uses linear algebra to transform the dataset into a compressed form. We can implement PCA feature selection technique with the help of PCA class of scikit-learn Python library. We can select number of principal components in the output.
#Example
#In this example, we will use PCA to select best 3 Principal components from Pima Indians Diabetes dataset.
from pandas import read_csv
from sklearn.decomposition import PCA
path = r'diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values
#Next, we will separate array into input and output components −
X = array[:,0:8]
Y = array[:,8]
#The following lines of code will extract features from dataset −
pca = PCA(n_components = 3)
fit = pca.fit(X)
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)
#Output
#Explained Variance: [ 0.88854663 0.06159078 0.02579012]
#[[ -2.02176587e-03 9.78115765e-02 1.60930503e-02 6.07566861e-02
#9.93110844e-01 1.40108085e-02 5.37167919e-04 -3.56474430e-03]
#[ 2.26488861e-02 9.72210040e-01 1.41909330e-01 -5.78614699e-02
#-9.46266913e-02 4.69729766e-02 8.16804621e-04 1.40168181e-01]
#[ -2.24649003e-02 1.43428710e-01 -9.22467192e-01 -3.07013055e-01
#2.09773019e-02 -1.32444542e-01 -6.39983017e-04 -1.25454310e-01]]
#We can observe from the above output that 3 Principal Components bear little resemblance to the source data.


#########################################Feature Importance
#As the name suggests, feature importance technique is used to choose the importance features. It basically uses a trained supervised classifier to select features. We can implement this feature selection technique with the help of ExtraTreeClassifier class of scikit-learn Python library.
#Example
#In this example, we will use ExtraTreeClassifier to select features from Pima Indians Diabetes dataset.
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
path = r'diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values
#Next, we will separate array into input and output components −
X = array[:,0:8]
Y = array[:,8]
#The following lines of code will extract features from dataset −
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)