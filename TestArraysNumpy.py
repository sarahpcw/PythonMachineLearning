import numpy as np 
a = np.arange(4420) * 0.98  # apply the power of 2 to every element
print(a)     	#[ 0  1  4  9 16 25 36 49 64 81]
print(a[3]) 	# prints only the 4th element (start counting at 0)
    			#9
print(a[3:5]) 	# prints 4th and 5th element not the 6th i.e. 5  (includes start, excludes end)
    			 #[ 9 16]
a[0]=1000        # changes the value in the 0 position to 1000
print('1',a) 
    			#1 [1000    1    4    9   16   25   36   49   64   81]
a[0:2:1]=1000 	# start : stop(exclusive the last one) : step
print('2',a) 
    			#2 [1000 1000    4    9   16   25   36   49   64   81]
a[0:8:2]=1000 	# start : stop(exclusive the last one) : step
print('3',a) 	#3 [1000 1000 1000    9 1000   25 1000   49   64   81]
print('4',a[::-1]) 	# prints the reverse of a
    			#4 [  81   64   49 1000   25 1000    9 1000 1000 1000]
print('_____________________')
print('3',a) 
print('4',a[::-3]) 	# every third [  81 1000    9 1000]
print('4',a[::-2]) 	# every second from the end [  81   49   25    9 1000]
print('4',a[0:-7]) 	# drop 7 and show the rest
print('4',a[0:7]) 	# drop 7 and show the rest

a = a.reshape(442,10)
a=a[:, np.newaxis, 2]
print (a.shape)
diabetes_X_train = a[:-20]
diabetes_X_test = a[-20:]

a1 = np.arange(100) * 0.98
a1 = a1.reshape(10,10)
print ( a1[:,:])
a1=a1[:, np.newaxis, 2]
print (a1.shape)
print ('a1')
print (a1)
print ('a1[:-2]')
diabetes_X_train = a1[:-2]
print ( diabetes_X_train)
print ('a1[-2:]')
diabetes_X_test = a1[-2:] 
print ( diabetes_X_test  )

 
#
## Split the targets into training/testing sets
#diabetes_y_train = diabetes.target[:-20]
#diabetes_y_test = diabetes.target[-20:]