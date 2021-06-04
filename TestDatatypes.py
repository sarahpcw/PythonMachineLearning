user_input = "0"
b = '0'
if type(user_input)==int:
    print ('user_input is a number')
if type(user_input)==str:
    print ('user_input is a string')
    
count = 0
while (count < 3 ) :
    count += 1
    user_input = input ('Enter a number: ')  
    if  ( user_input.isnumeric() )  :
        user_input = int(user_input)
        break
print ('broken',type(user_input) )    

count = 0
while (count < 3 and type(user_input)!=int) :
    count += 1
    try:
        user_input = int ( input ('Enter a number: ')  )
    except:
        print('Invalid input')
    else:
        print ('The value given is ',user_input)
    finally: 
        print ('finally')






#import math
#import numpy as np
#nr = [8,5,6]
#a = np.average(nr)
#print ('---- ', a)
#print  ( np.amin(nr) ) 
##Return the minimum of an array or minimum along an axis.
#print ( np.amax(nr)
##Return the maximum of an array or maximum along an axis.
#
##print ( np.nanmin(nr ) )
##Return minimum of an array or minimum along an axis, ignoring any NaNs.
#print ( np.nanmax(nr ) )
##Return the maximum of an array or maximum along an axis, ignoring any NaNs.
#print ( np.ptp(nr) ) 
##Range of values (maximum - minimum) along an axis.
#print ( np.percentile(nr))
#
##Compute the qth percentile of the data along the specified axis.
#print ( np.nanpercentile(nr) )
##Compute the q
