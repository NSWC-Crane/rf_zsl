#a program to go through given weights and update them to new weights based on a set value

import numpy
from numpy import float32

def convert_weights(array):

    divisor = 780#780#780#780#780#780#780#290#370

    arr = array[:]
    shape = numpy.shape(arr)
    print(shape)
    #i = shape[0]
    #j = shape[1]
    i =0
    j =0
    new = array[:].astype(int)

    while(i < shape[0]):
       # print(i)
        j=0
        while(j < shape[1]):
            #print(j)
            #print(arr[i][j])
            #arr[i][j] = 66
            #print(arr[i][j])
            index = 1
            prev = 0
            curr = 0
            boo = False
            boo2 = True

            while(boo2):
                y = arr[i][j]
                if (y < 0):
                    #print("testing1")
                    #index = index * -1
                    boo = True
                    y = y*-1
                #print("testing2")
                prev = curr
                curr = index / divisor
                #print(prev,y,curr)
                if(prev < y < curr):
                    #print(index , i, j)
                    l = [prev, curr]
                    update = min(l, key = lambda x:abs(x-y))
                    #print("testing")
                    #print(i,j)
                    arr[i][j] = update
                    #new[i][j] = index
                    if(boo):
                        #new[i][j] = index * -1                             #bug in new array is need to distinuigh high vs low index
                        arr[i][j] = arr[i][j] * -1
                        if(update == 0):
                            arr[i][j] = -1/divisor
                    if(update == 0 and not boo):
                        arr[i][j] = 1/divisor
                    boo2 = False
                #print("testing2")
                index = index + 1
            j = j+1
        i = i+1


    print ("done")

    return arr



x = numpy.array([[ 0.00232184, -0.5693014 , -0.5884882 , -0.03769392,  0.59167534,
        -0.08084819, -0.59842384,  0.59088916],
       [-0.47640613,  0.14825755,  0.48201182, -0.16015607, -0.09390862,
         0.04388316, -0.11757955, -0.19412404],
       [-0.31293496, -0.4530124 ,  0.4653467 ,  0.1483317 ,  0.23737249,
         0.60953546, -0.33516   , -0.16978475],
       [-0.3509881 ,  0.45543545, -0.09724227,  0.01627445, -0.484733  ,
         0.30799463, -0.38777274,  0.16850866],
       [ 0.44842452, -0.08754146,  0.08668187,  0.07608068, -0.56855613,
         0.21641049, -0.29942024,  0.5664803 ],
       [ 0.31708813, -0.58601314, -0.04750733, -0.40087062,  0.5742284 ,
        -0.61173576,  0.22690785,  0.04606936],
       [-0.39570424, -0.23053056,  0.06598631, -0.36287755,  0.2695459 ,
         0.3632156 , -0.48130488, -0.1019057 ],
       [ 0.26382202,  0.08474493,  0.4831146 , -0.50381887,  0.5101871 ,
        -0.42354593,  0.11591077,  0.36066732]], dtype=float32)
print(x)
y =convert_weights(x) 


#print(x)
print(y)
#print(z)
