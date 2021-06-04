# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 22:43:23 2021

@author: u
"""

import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area   = (30 * np.random.rand(N))**2  # 0 to 15 point radii
print(area)
plt.scatter( x, y, s=area, c=colors, alpha=0.5 ) #alpha transparency, s is the size of the dots
plt.show()
