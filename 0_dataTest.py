import pandas as pd
import numpy as np
df = pd.DataFrame(np.rint(np.random.randn(5,3)),index=['a','c','e','g','i'],columns = ['one','two','three'])
print(df)
dfBackup = df  
#
#print ('create 80% frame')
df80 = df.sample(frac=0.8)
print (df80)
#print (df80.shape)
#
#print ('create 20% frame')
df20 = df.drop(df80.index)
print (df20)
#print (df80.shape)
#
##If we merely want to remove random rows we can use drop and the inplace parameter:
#print (df.shape)
#print (df20.index)
df = df.drop(df20.index)
#print (df)
#print (df.shape)
#
#df = dfBackup
##Same as: 
#df2 = df.drop(df.sample(frac=0.8).index)
## Output: (3909, 5)
#print ('df2')
#print (df2)
#print ('col')
#print ('col', df.columns)
# 
