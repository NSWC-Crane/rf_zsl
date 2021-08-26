import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow import keras
from numpy import float32

import tensorflow_model_optimization as tfmot
from zsl_error_metric import *

from sort_weights import rewrite_weights, binary
 
class ZPL(Model):
    def __init__(self):
        super(ZPL, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Dense(units = 8, activation='relu', use_bias = False),
            layers.Dense(units = 16, activation='relu', use_bias = False), 
            layers.Dense(units = 128, activation='relu', use_bias = False), 
            layers.Dense(units = 1, activation='relu', use_bias = False),    
            ])

        self.decoder = tf.keras.Sequential([  
            #layers.Dense(units = 8, use_bias = False,trainable=True),
            #layers.Dense(units = 500, use_bias = False,trainable=True),
            layers.Dense(units = 2048, use_bias = False,trainable=True),
            ])


    def call(self, x):
        encoder = self.encoder(x)
        decoded = self.decoder(encoder)
        return decoded


        #not sure if following will work
    def freeze_decoder(self):
        self.decoder.layers[0].trainable = False
        #self.decoder.layers[1].trainable = False
        #self.decoder.layers[2].trainable = False

    def get_feature(self,x):
        return self.encoder(x)


class cluster(Model):
    def __init__(self, y):
        super(cluster, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Dense(units = 8, activation='relu', use_bias = False),
            layers.Dense(units = 16, activation='relu', use_bias = False), 
            layers.Dense(units = 128, activation='relu', use_bias = False),
            layers.Dense(units = 1, activation='relu', use_bias = False),
            ])

        self.decoder = y


    def call(self, x):
        encoder = self.encoder(x)
        decoded = self.decoder(encoder)
        return decoded


        #not sure if following will work
    def freeze_decoder(self):
        self.decoder.layers[0].trainable = False
        #self.decoder.layers[1].trainable = False
        #self.decoder.layers[2].trainable = False



model = ZPL()
model.build((None, 2048))
#print(model.layers[1].layers[0].trainable)
#model.freeze_decoder()
#print(model.layers[1].layers[0].trainable)







fp_bits = 4         # number of bits used to represent the weights
fp_range =2**fp_bits      # the max value
# the min/max number (0 <= x < fp_range)
fp_min = 0
fp_max = fp_range - 1
scale = 12.6 * (fp_range - 1)    # this is the scale factor to divide the final number by

#scale = 15
#setting weights manually
rnd_range = 1/fp_range
mr = np.random.default_rng(10)


#want to init weights to random values
#x = mr.integers(fp_min, fp_max, [5,150]).astype(np.float32)/(scale)
#y = mr.integers(fp_min, fp_max, [150,500]).astype(np.float32)/(scale)
z = mr.integers(fp_min, fp_max, [1,2048]).astype(np.float32)/(scale)
model.layers[1].set_weights([z])#x,y,z])

t = mr.integers(fp_min, fp_max, [2048,8]).astype(np.float32)/(scale)
g = mr.integers(fp_min, fp_max, [8,16]).astype(np.float32)/(scale)
h = mr.integers(fp_min, fp_max, [16,128]).astype(np.float32)/(scale)
f = mr.integers(fp_min, fp_max, [128,1]).astype(np.float32)/(scale)
model.layers[0].set_weights([t,g,h,f])


input_size=2048
idx = 10000
rng = np.random.default_rng()
#read data
#x = np.fromfile("../data/dataset/lfm_test_10M_100m_0000.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)
xd = np.fromfile("../data/dataset/VH1-164.sigmf-data.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)
x = xd[np.s_[idx:idx+input_size]]
#scale because cant have negative numbers
x = x +4096
X = tf.convert_to_tensor(x)
X = tf.reshape(X, [-1, 2048])

idx = 4096
y = xd[np.s_[idx:idx+input_size]]
#scale because cant have negative numbers
y = y +4096
Y = tf.convert_to_tensor(y)
Y = tf.reshape(Y, [-1, 2048])




#loop through entire file



opt = tf.keras.optimizers.Adam(learning_rate=5e-9)
opt2 = tf.keras.optimizers.Adam(learning_rate=5e-4)
opt3 = tf.keras.optimizers.Adam(learning_rate=5e-4)






#skipping model.round_weights(128)
#the scale factor?


callback2 = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=.01, patience=30)




model.compile(optimizer=opt, loss=losses.MeanSquaredError())
history = model.fit(X,X, epochs=100000000, callbacks=[callback2])

model.freeze_decoder()


nc = 7


np.set_printoptions(threshold=np.inf)


print(model.layers[1].weights)



print("before clustering")
cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

clustering_params = {
        'number_of_clusters': nc,
        'cluster_centroids_init': CentroidInitialization.LINEAR #DENSITY_BASED
    }

clustered_model = cluster_weights(model.decoder, **clustering_params)
#clustered_model.compile(optimizer=opt, loss=losses.MeanSquaredError())

model2 = cluster(clustered_model)

model2.build((None, 2048))
model2.layers[0].set_weights([t,g,h,f])


model2.compile(optimizer=opt2, loss=losses.MeanSquaredError())


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=.0001, patience=100)

print(model2.layers[1].trainable)
model2.layers[1].trainable = False

#np.set_printoptions(threshold=np.inf)

print(model2.layers[1].trainable)

#print(model.layers[1].weights)

#print(model2.layers[1].trainable)
#print(model2.layers[1].weights)
history2 = model2.fit(X,X,epochs =10000000, callbacks=[callback])




#rewrite weights / retrain for compression
#temp_array = tf.make_ndarray(model2.layers[1].weights)
#print(model2.layers[1].get_weights)
#temp_array = model2.layers[1].get_weights


temp_array = []
for weight in model2.layers[1].get_weights():
    temp_array.append(weight)

temp_array = np.reshape(temp_array, [2048])
#temp_array = temp_array.astype(np.float32)
new_weights, compressed_weights, temp_sign, temp_div,temp_clusters, cluster_list  = rewrite_weights(temp_array, nc)
#might need to reshape here
#new_weights = new_weights.astype(np.float32)
new_weights = tf.convert_to_tensor(new_weights)
new_weights = tf.reshape(new_weights, [-1, 2048])
#print(new_weights)
new_weights = tf.cast(new_weights, dtype=np.float32)
#print(new_weights)
model3 = ZPL()
#model3.layers[1].set_weights = new_weights

#print(model3.layers[1].weights)
#print(new_Weights)

model3.build((None, 2048))
model3.layers[1].set_weights([new_weights])
model3.layers[0].set_weights([t,g,h,f])
model3.layers[1].trainable = False
print(model3.layers[1].weights)
print(new_weights)

model3.compile(optimizer = opt3, loss=losses.MeanSquaredError())
history3 = model3.fit(X,X, epochs=100000000, callbacks=[callback])

Y = model3.predict(X)

print(X-Y)

t2 = Y
t2 = t2.reshape(2048).astype(np.int16)
t2 = t2 - 4096


zsl_metric = zsl_error_metric(X.numpy(), Y)
print(zsl_metric)




#write the weights to file

#compressed_weights, temp_sign, temp_div, temp_clusters, cluster_list

#print (compressed_weights, temp_sign)


counter = 0
string_of_bits = ""

cluster_array = {cluster_list[0]:0, 
                 cluster_list[1]:1, 
                 cluster_list[2]:2,
                 cluster_list[3]:3,
                 cluster_list[4]:4,
                 cluster_list[5]:5,
                 cluster_list[6]:6}

print(cluster_array.get(cluster_list[1]))
print(cluster_list)
while(counter < 2048):


    weight_string = ""

    #cluster number
    cluster_number = cluster_array.get(temp_clusters[counter])
    cluster_bin = '{0:03b}'.format(cluster_number)
    weight_string += cluster_bin
    
    #sign
    if(temp_sign[counter]):
        weight_string += "1"
    else:
        weight_string += "0"
    
    #div
    if(temp_div[counter]):
        weight_string += "1"
    else:
        weight_string += "0"

    #weight
    #print(compressed_weights[counter])
    weight_string += '{0:07b}'.format(abs(compressed_weights[counter]))
    #print(weight_string)

    string_of_bits += weight_string

    counter = counter +1


print(string_of_bits)
print(len(string_of_bits))



bit_feature = binary(model3.get_feature(X))
print(bit_feature)

import struct

counter = 0
while counter < 17 :

    feature_string = bit_feature[counter:counter+16] 
    
    print(feature_string)

    with open("test.bnr", "ab") as f:
        f.write(struct.pack('i', int(feature_string[::-1], 2)))

    counter = counter +16

#print the clusters to the file

counter =0
while(counter < len(cluster_list)):
    
    bit_string = binary(cluster_list[counter])

    counter2 = 0
    while counter2 < 17 :

        bit_string = bit_feature[counter2:counter2+16]

        with open("test.bnr", "ab") as f:
            f.write(struct.pack('i', int(bit_string[::-1], 2)))
        counter2 = counter2+16
    counter = counter +1





counter =0
while(counter < len(string_of_bits)):
    
    current = string_of_bits[counter:(counter +31)]
    #print( current)

    #if(len(current) < 31):

    with open("test.bnr", "ab") as f:
        f.write(struct.pack('i', int(current[::-1], 2)))


    counter = counter + 31




#np.set_printoptions(threshold=np.inf)





print("done")




