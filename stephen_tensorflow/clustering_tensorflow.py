import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow import keras
from numpy import float32

import tensorflow_model_optimization as tfmot

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
idx = 0
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



opt = tf.keras.optimizers.Adam(learning_rate=5e-4)
opt2 = tf.keras.optimizers.Adam(learning_rate=5e-4)






#skipping model.round_weights(128)
#the scale factor?


callback2 = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=.1, patience=20)




model.compile(optimizer=opt, loss=losses.MeanSquaredError())
history = model.fit(X,X, epochs=10000, callbacks=[callback2])

model.freeze_decoder()


nc = 2

cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

clustering_params = {
        'number_of_clusters': nc,
        'cluster_centroids_init': CentroidInitialization.DENSITY_BASED
    }

clustered_model = cluster_weights(model.decoder, **clustering_params)
#clustered_model.compile(optimizer=opt, loss=losses.MeanSquaredError())


model2 = cluster(clustered_model)

model2.build((None, 2048))
model2.layers[0].set_weights([t,g,h,f])


model2.compile(optimizer=opt2, loss=losses.MeanSquaredError())


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=.0001, patience=300)

print(model2.layers[1].trainable)
model2.layers[1].trainable = False
#print(model2.layers[1].trainable)
#print(model2.layers[1].weights)
history2 = model2.fit(X,X,epochs =10000000, callbacks=[callback])
#print(model2.layers[1].weights)

#history3 = model2.fit(Y,Y,epochs =10000000, callbacks=[callback])

print("done")





