import numpy

import struct
def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))


def rewrite_weights(weights,number_of_clusters = 7):


    x = weights


    #print(x)

    counter =1

    x1 = min(x)
    x7 = max(x)
    
    first = distance = x1 + (1/(number_of_clusters -1))*(x7-x1)
    curr = round(first, 8)
    prev2 = x1
    prev = x1

    #a_list is the list of clusters
    a_list = []
    a_list.append(x1)   #x1
    a_list.append(curr) #x2

    
    #x1 = min(x)
    #x7 = max(x)
    #x2 = round(distance,8)
    #x3 = round(x2 + (x2-x1),8)
    #x4 = round(x3 + (x3-x2),8)
    #x5 = round(x4 + (x4-x3),8)
    #x6 = round(x5 + (x5-x4),8)

    #find the clusters at = intervals (assuming we are linear based)
    while counter < number_of_clusters-2 :

    
        prev2 = prev
        prev = curr 
        curr = round(prev + (prev - prev2), 8)
        a_list.append(curr)
        counter = counter +1

    a_list.append(x7)

    #a_list = [x1,x2,x3,x4,x5,x6,x7]
    print(a_list)
    temp_values = []   #compressed values
    temp_clusters = [] #to know which weight belongs to which cluster
    temp_div = []      #do we multiply/divide by 10 or 100
    temp_sign = []     #sign for each weight when adding/sub back to cluster
    final_values = []  #rebuilt after compressed

    
    temp = [1,2,3,4,5,6]


    print(x)
    #for j in x:
    for i in x:
        given_value = i
        absolute_difference_function = lambda list_value : abs(list_value - given_value)
        
        #find the closest cluster and make it a positive int
        weight_value = round(i - min(a_list, key = absolute_difference_function),8)*100000000
        
        #add compressed int to weight values
        temp_values.append(int(weight_value/100))
        print(weight_value)
        
        #which cluster it belongs to
        temp_clusters.append(min(a_list, key = absolute_difference_function))
        cluster = min(a_list, key = absolute_difference_function)
        if(i - cluster < 0):
            temp_sign.append(True)
        else:
            temp_sign.append(False)

    #print(temp)

    #do we divide by another 10 because number is too large
    counter = 0
    for i in temp_values:
        if(abs(i) > 99):
            temp_values[counter] = int(i/10)
            temp_div.append(True)
        else:
            temp_div.append(False)
        counter = counter + 1
    print(temp_values)

    compressed_weights = temp_values.copy()
    print(compressed_weights)
    print(len(temp_values), len(temp_clusters), len(temp_div), len(temp_sign))

    
    #rebuild the compressed weights (there is loss in the weights)
    max_len = 32000
    for i in range(0,2048): #ijkl, temp_values, temp_clusters, temp_div, temp_sign:
        value = temp_values[i]
        if False: #temp_div[i]:
            #need to mulitply by 10,000
            temp_values[i] = temp_values[i] * 1000
            if temp_sign[i]:
                value = temp_clusters[i] - (temp_values[i] / 100000000)
            else:
                value = temp_clusters[i] + (temp_values[i] / 100000000)
        else:
            temp_values[i] = temp_values[i] * 100

            if temp_sign[i]:
                value = temp_clusters[i] - (temp_values[i] / 100000000)
            else:
                value = temp_clusters[i] + (temp_values[i] / 100000000)
        final_values.append(value)

    print(compressed_weights)

    return final_values , compressed_weights, temp_sign, temp_div, temp_clusters, a_list


