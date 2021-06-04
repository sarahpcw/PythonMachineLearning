"""\
from sklearn import datasets
load_boston()          Load and return the boston house-prices dataset (regression).
load_iris()            Load and return the iris dataset (classification).
load_diabetes()        Load and return the diabetes dataset (regression).
load_digits([n_class]) Load and return the digits dataset (classification).
load_linnerud()        Load and return the linnerud dataset (multivariate regression).

@"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
print ('type', type (diabetes))
 
    # Make sure the embedding works by default.
#data = datasets.load_diabetes()
print (diabetes.keys()) #dict_keys(['data', 'target', 'DESCR', 'feature_names'])

print (type ( diabetes['feature_names']) )
print (  diabetes['feature_names'] ) 
print ( diabetes['DESCR'] )

X = diabetes['data']  # this is an array slice with indexes
y = diabetes['target']
print (X.shape) # (442, 10)
print (y.shape) # (422,)
print ( diabetes['DESCR'] )

from pandas import read_csv
path = r"diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#data = read_csv(path, names=headernames)
data = read_csv(path)  # comes in as a pandas dataframe slice with loc and iloc
print(data.head(0))
print(data.dtypes)
#preg     object
#plas     object
#pres     object
#skin     object
#test     object
#mass     object
#pedi     object
#age      object
#class    object
#dtype: object
print(data.describe())
#       preg plas pres skin test mass   pedi  age class
#count   769  769  769  769  769  769    769  769   769
#unique   18  137   48   52  187  249    518   53     3
#top       1  100   70    0    0   32  0.254   22     0
#freq    135   17   57  227  374   13      6   72   500

#Reviewing Class Distribution
#Class distribution statistics is useful in classification problems where we need to know the 
#balance of class values. It is important to know class value distribution because if we have highly 
#imbalanced class distribution i.e. one class is having lots more observations than other class,
# then it may need special handling at data preparation stage of our ML project. 
# We can easily get class distribution in Python with the help of Pandas DataFrame.
#Example

count_class = data.groupby('class').size()
print(count_class)
#class
#0          500
#1          268
#Outcome      1
#dtype: int64

#Reviewing Correlation between Attributes
#The relationship between two variables is called correlation. In statistics, the most common method for calculating correlation is Pearson’s Correlation Coefficient. It can have three values as follows −
#Coefficient value = 1 − It represents full positive correlation between variables.
#Coefficient value = -1 − It represents full negative correlation between variables.
#Coefficient value = 0 − It represents no correlation at all between variables.
#It is always good for us to review the pairwise correlations of the attributes in our dataset
# before using it into ML project because some machine learning algorithms such as linear regression
# and logistic regression will perform poorly if we have highly correlated attributes. In Python,
# we can easily calculate a correlation matrix of dataset attributes with the help of corr() 
# function on Pandas DataFrame.
#Example

from pandas import read_csv
from pandas import set_option
path = r"diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=names)
print ('SKEW:')
print(data.skew())

print ('set options:')
set_option('display.width', 50)
set_option('precision', 2)
correlations = data.corr(method='pearson')
print(correlations)


from pandas import read_csv
path = r"iris.csv"
data = read_csv(path)
print(data.shape)
print(data[:3])
print ('SKEW:')
print(data.skew())

print ('set options:')
set_option('display.width', 100)
set_option('precision', 2)
correlations = data.corr(method='pearson')
print(correlations)

print ('set options:')
set_option('display.width', 30)
set_option('precision', 2)
correlations = data.corr(method='pearson')
print(correlations)

x = np.zeros((3,5))  # 3 down 5 accross
print (x[:, 0:1])
print (x[:, 0:1].shape)
print (x[:, 0:2])
print (x[:, 0:2].shape)
print (x[:, 0:3])
print (x[:, 0:4].shape)
print (x[:, 0:4])
print (x[:, 0:4].shape)
print (x[:, 0:5])
print (x[:, 0:5].shape)  # what is the shape of 
print (x[:, np.newaxis,:])  # adds another dimension
print (x[:, np.newaxis,:].shape)
#print (np.zeros((3,5))[:,np.newaxis,:].shape)
# shape will be (3,1,5)