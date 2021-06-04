# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 22:50:29 2019

@author: u
"""

import csv
somedict = dict(raymond='red', rachel='blue', matthew='green')
print (somedict)


#with open('mycsvfile.csv','wb') as f:
#  w = csv.writer(f)
#  w.writerows(somedict.items())
#  
line=""
from sklearn import datasets 
CSV=""
diabetes = datasets.load_diabetes() # Load the diabetes dataset : 4 items: data, target, description, feature names
print ('type', type (diabetes))
#print ('diabetes.items() !! \n',diabetes.items())
count=0
for k,v in diabetes.items():
    print ('COUNT!!!',count, 'k = key !!!', k, type(v))
    print ('Value !!!', v)
    count+=1
#    line = "{},{}\n".format(k, ",".join(v))
#    CSV+=line
print('count ',count)
#You can store this CSV string variable to file as below
with open("filename.csv", "w") as file:
    file.write(CSV)
with open("filename.csv",'r') as f:
    reader = csv.reader( f , delimiter = ',' )
    data = list(reader)
print (data)

import pandas as pd 
df = pd.DataFrame.from_dict(diabetes, orient="index")
df.to_csv("data.csv") #Create a CSV file: