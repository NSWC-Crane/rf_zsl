import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow import keras
from numpy import float32

import math
import os
import datetime

import tensorflow_model_optimization as tfmot
from zsl_error_metric import *

class ZPL(Model):
    def __init__(self):
        super(ZPL, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Dense(units = 8, activation='relu', use_bias = False),
            layers.Dense(units = 16, activation='relu', use_bias = False), 
            layers.Dense(units = 128, activation='relu', use_bias = False), 
            layers.Dense(units = 1, activation='relu', use_bias = False),    
            #layers.Dense(units = 128, activation='relu', use_bias = False), 
            #layers.Dense(units = 32, activation='relu', use_bias = False), 
            #layers.Dense(units = 5, activation='relu', use_bias = False), 
            
            ])

        self.decoder = tf.keras.Sequential([  

            #layers.Dense(units = 8000, use_bias = False,trainable=True)
            layers.Dense(units = 8, use_bias = False,trainable=True),
            layers.Dense(units = 64, use_bias = False,trainable=True),
            layers.Dense(units = 32000, use_bias = False,trainable=True),
            ])


    def call(self, x):
        encoder = self.encoder(x)
        decoded = self.decoder(encoder)
        return decoded


        #not sure if following will work
    def freeze_decoder(self):
        self.decoder.layers[0].trainable = False
        self.decoder.layers[1].trainable = False
        self.decoder.layers[2].trainable = False

class cluster(Model):
    def __init__(self, y):
        super(cluster, self).__init__()

        self.encoder = tf.keras.Sequential([
            
            #layers.Dense(units = 128, activation='relu', use_bias = False), 
            #layers.Dense(units = 32, activation='relu', use_bias = False),
            #layers.Dense(units = 5, activation='relu', use_bias = False),
            
            
            
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
        self.decoder.layers[1].trainable = False
        self.decoder.layers[2].trainable = False



#model = ZPL()
#model.build((None, 2048))
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
p = mr.integers(fp_min, fp_max, [1,8]).astype(np.float32)/(scale)
l = mr.integers(fp_min, fp_max, [8,64]).astype(np.float32)/(scale)
m = mr.integers(fp_min, fp_max, [64,32000]).astype(np.float32)/(scale)
#m = mr.integers(fp_min, fp_max, [5,8000]).astype(np.float32)/(scale)
#model.layers[1].set_weights([x,y,z])

ty = mr.integers(fp_min, fp_max, [32000,8]).astype(np.float32)/(scale)
gy = mr.integers(fp_min, fp_max, [8,16]).astype(np.float32)/(scale)
hy = mr.integers(fp_min, fp_max, [16,128]).astype(np.float32)/(scale)
fy = mr.integers(fp_min, fp_max, [128,1]).astype(np.float32)/(scale)
#model.layers[0].set_weights([t,g,h,f])


input_size=32000
idx = 0
rng = np.random.default_rng()
#read data
#xd = np.fromfile("../data/dataset/lfm_test_10M_100m_0000.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)
xd = np.fromfile("../data/dataset/VH1-164.sigmf-data.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)
#x = x[np.s_[idx:idx+input_size]]
#scale because cant have negative numbers
#x = x +4096
#X = tf.convert_to_tensor(x)
#X = tf.reshape(X, [-1, 2048])




io_size=32000
x_blocks = math.ceil(xd.size/io_size)
data_type = "sdr"
base_name = "sdr_test"
date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
scenario_name = "fs{:02d}-io{:03d}-".format(5, 2048) + data_type
log_dir = "../results/" + scenario_name + "/"

os.makedirs(log_dir, exist_ok=True)

# writer to save the data
test_writer = open((log_dir + 'io32000_clustering32' + "VH1-proceessed_" + "data.bin"), "ab")
log_writer = open('./log32000_feature1_cluster32.txt', "a")

i = 1


callback2 = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=.1, patience=20)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=.0001, patience=300)


cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

nc = 32

clustering_params = {
    'number_of_clusters': nc,
    'cluster_centroids_init': CentroidInitialization.LINEAR
}

#opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
#opt2 = tf.keras.optimizers.Adam(learning_rate=5e-4)

counter = 0
print("Processing...\n")
for idx in range(0, x_blocks*io_size, io_size):
#while(counter <1):
    #idx = 7749*2048
    counter = counter + 1
    print("running idx: {}\n\n\n".format(idx))

    #counter = 0
    #check = False
    #while(counter < 5 or not check):

    f = open('./file.txt','a')
    print("{}\n".format(i), file=f)
    f.close()

    i = i+1

    x = xd[idx:(idx + io_size)]
    print(x)
    print("test")
    # get the mean of x
    #x_mean = math.floor(np.mean(x))
    #x_std = np.std(x)

    x = x +4096
    X = tf.convert_to_tensor(x)
    #X = tf.reshape(X, [-1, 2048])

    if (x.size < io_size):
        X = tf.pad(X,[[0,io_size-x.size]], 'CONSTANT')

    X = tf.reshape(X, [-1, 32000])




    opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
    opt2 = tf.keras.optimizers.Adam(learning_rate=5e-4)


    



#skipping model.round_weights(128)
#the scale factor?


    model = ZPL()
    model.build((None, 32000))

    model.layers[1].set_weights([p,l,m])
    model.layers[0].set_weights([ty,gy,hy,fy])


    #callback2 = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=.1, patience=20)

    model.compile(optimizer=opt, loss=losses.MeanSquaredError())
    history = model.fit(X,X, epochs=100000, callbacks=[callback2])

    model.freeze_decoder()


    #nc = 32

    #cluster_weights = tfmot.clustering.keras.cluster_weights
    #CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

    #clustering_params = {
    #    'number_of_clusters': nc,
    #    'cluster_centroids_init': CentroidInitialization.LINEAR
    #}

    clustered_model = cluster_weights(model.decoder, **clustering_params)
    #clustered_model.compile(optimizer=opt, loss=losses.MeanSquaredError())


    model2 = cluster(clustered_model)

    model2.build((None, 32000))
    model2.layers[0].set_weights([ty,gy,hy,fy])
    model2.layers[1].trainable = False

    model2.compile(optimizer=opt2, loss=losses.MeanSquaredError())


    #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=.0001, patience=300)


    history2 = model2.fit(X,X,epochs =10000000, callbacks=[callback])

    Y = model2.predict(X)

        #Z = X - Y
        #Z = Z.reshape([z.size]).astype(np.int16)
        #if(Z.item(6) < 5):
        #    check = True
        #counter = counter + 1
        #check = True

    #need to save the finished data
    t2 = Y
    t2 = t2.reshape(32000).astype(np.int16)
    t2 = t2 - 4096
    
     
    zsl_metric = zsl_error_metric(X.numpy(), Y)
    log_writer.write("zsl_metrics_index{}: {}\n".format(idx, ",".join(str(x) for x in zsl_metric)))
    test_writer.write(t2)

test_writer.close()
log_writer.close()



print("done")





