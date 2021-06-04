# -*- coding: utf-8 -*-
"""
Print diabetes

simply load and plot the data 

"""

print ( 'scatter mix plot' )

#from matplotlib import pyplot
#from pandas import read_csv
#path = r"diabetes.csv"
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#data = read_csv(path, names=names)
#data.hist()
#pyplot.show()



#from pandas.tools.plotting import scatter_matrix
#path = r"diabetes.csv"
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#data = read_csv(path, names = names)
#scatter_matrix(data)
#pyplot.show()


from matplotlib import pyplot
from pandas import read_csv
import numpy
Path = r"diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(Path, names = names)
correlations = data.corr()
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()