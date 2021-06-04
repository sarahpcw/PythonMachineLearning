# -*- coding: utf-8 -*-
"""
Test the amzon data
@author: u
"""

# -*- coding: utf-8 -*-

"""

Created on Fri Dec 21 18:59:49 2018



@author: Nhan Tran

"""



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Importing the dataset
dataset = pd.read_csv('https://s3.us-west-2.amazonaws.com/public.gamelab.fun/dataset/position_salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

print (dataset)