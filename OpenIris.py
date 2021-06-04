import numpy as np
import matplotlib.pyplot as pl
x = np.linspace(-np.pi,np.pi,256,endpoint=True) 
y = np.linspace((-np.pi-100),(np.pi+100),256,endpoint=True) 
#generate 256 numbers between -3.14 and +3.14
#print (x)
#cp = np.cos(x)
#sp = np.sin(x)
pl.scatter(x,y)
#pl.scatter(x,sp)
#pl.show()
