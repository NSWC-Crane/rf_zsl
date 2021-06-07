import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow import keras


from numpy import float32
#from weights import set_encoder_weights
from weights_f5 import set_decoder_weights
from weights_f4 import set_decoder_weights_f4
from weights_f6 import set_decoder_weights_f6
from weights_f5_input2048 import set_decoder_f5_input2048
from update_weights import convert_weights
from encoder_weights import set_encoder_weights_f5

import matplotlib.pyplot as plt


printing = False
graphing = True


input_size = 8
feature_size = 8
decoder_int1 = 4

read_data = True

max_epochs = 26000

x_plot = []
y_plot = []

output_array = []

#main model
#layers.Linear DNE
class ZPL(Model):
  def __init__(self):
    super(ZPL, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(units = 8, activation='relu', use_bias = False), #was DNE
      layers.Dense(units = 16, activation='relu', use_bias = False), #was 16
      layers.Dense(units = 128, activation='relu', use_bias = False),  #was 8
      layers.Dense(units = 5, activation='relu', use_bias = False),    #was DNE
      ])


    #layer1 = keras.layers.Dense(8, activation='relu')
    #layer1.trainable = False
    self.decoder = tf.keras.Sequential([  #keras.Input(shape=(8,)), layer1])
      layers.Dense(units = 8, activation='relu', use_bias = False,trainable=False), #was DNE
      layers.Dense(units = 64, activation='relu', use_bias = False,trainable=False),   #was 8
      layers.Dense(units = 2048, activation='relu', use_bias = False,trainable=False),  #was 16
      #layers.Dense(units = 8, activation='relu', use_bias = False,trainable=False),  #was DNE
      ])



  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class ZPL_one(Model):
  def __init__(self):
    super(ZPL_one, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(units = 256, activation='relu', use_bias = False), #was DNE
      layers.Dense(units = 64, activation='relu', use_bias = False), #was 16
      layers.Dense(units = 32, activation='relu', use_bias = False),  #was 8
      layers.Dense(units = 5, activation='relu', use_bias = False),    #was DNE
      ])


    #layer1 = keras.layers.Dense(8, activation='relu')
    #layer1.trainable = False
    self.decoder = tf.keras.Sequential([  #keras.Input(shape=(8,)), layer1])
      layers.Dense(units = 8, activation='relu', use_bias = False,trainable=False), #was DNE
      layers.Dense(units = 64, activation='relu', use_bias = False,trainable=False),   #was 8
      layers.Dense(units = 2048, activation='relu', use_bias = False,trainable=False),  #was 16
      #layers.Dense(units = 8, activation='relu', use_bias = False,trainable=False),  #was DNE
      ])



  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class ZPL_two(Model):
  def __init__(self):
    super(ZPL_two, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(units = 1024, activation='relu', use_bias = False), #was DNE
      layers.Dense(units = 512, activation='relu', use_bias = False), #was 16
      layers.Dense(units = 64, activation='relu', use_bias = False),  #was 8
      layers.Dense(units = 5, activation='relu', use_bias = False),    #was DNE
      ])


    #layer1 = keras.layers.Dense(8, activation='relu')
    #layer1.trainable = False
    self.decoder = tf.keras.Sequential([  #keras.Input(shape=(8,)), layer1])
      layers.Dense(units =8, activation='relu', use_bias = False,trainable=False), #was DNE
      layers.Dense(units = 64, activation='relu', use_bias = False,trainable=False),   #was 8
      layers.Dense(units = 2048, activation='relu', use_bias = False,trainable=False),  #was 16
      #layers.Dense(units = 8, activation='relu', use_bias = False,trainable=False),  #was DNE
      ])



  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class ZPL_three(Model):
  def __init__(self):
    super(ZPL_three, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(units = 1024, activation='relu', use_bias = False), #was DNE
      layers.Dense(units = 2408, activation='relu', use_bias = False), #was 16
      layers.Dense(units = 512, activation='relu', use_bias = False),  #was 8
      layers.Dense(units = 5, activation='relu', use_bias = False),    #was DNE
      ])


    #layer1 = keras.layers.Dense(8, activation='relu')
    #layer1.trainable = False
    self.decoder = tf.keras.Sequential([  #keras.Input(shape=(8,)), layer1])
      layers.Dense(units = 8, use_bias = False,trainable=True), #was DNE
      layers.Dense(units = 2048, use_bias = False,trainable=True),   #was 8
      layers.Dense(units = 2048, use_bias = False,trainable=False),  #was 16
      #layers.Dense(units = 8, activation='relu', use_bias = False,trainable=False),  #was DNE
      ])



  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


counter = 0
y_plot_secret = []
y_plot_secret2 = []
np.set_printoptions(threshold=np.inf)

while(counter < 1):


    #if(counter ==0):
    #    autoencoder = ZPL()
    #if(counter ==1):
    #    autoencoder = ZPL_one()
    #if(counter ==2):
    #    autoencoder = ZPL_two()
    #if(counter ==3):
    #    autoencoder = ZPL_three()

    #autoencoder = ZPL_two()

    #autoencoder = ZPL()
    autoencoder = ZPL_three()
    autoencoder.build((None, 2048))

#device = "cpu"


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
    #x = mr.integers(fp_min, fp_max, [5,8]).astype(np.float32)/(scale)
    #y = mr.integers(fp_min, fp_max, [8,4096]).astype(np.float32)/(scale)
    #z = mr.integers(fp_min, fp_max, [4096,2048]).astype(np.float32)/(scale)
#this made redundant? #w = mr.integers(fp_min, fp_max, [16,8]).astype(np.float32)/(scale)

    from output_weights_norelu import set_decoder_norelu
    from weights_noreluall import set_decoder_noreluall #this is for 8->2048->2048 for decoder
    from weights_noreluall_1024_2048 import set_decoder_noreluall_1024_2048
    from weights_noreluall_4096_2048 import set_decoder_noreluall_4096_2048
    x,y,z = set_decoder_noreluall()
    #x = convert_weights(x)
    #y = convert_weights(y)
    #z = convert_weights(z)
    #w = convert_weights(w)

    #q,w,e,r = set_encoder_weights_f5()


#set the weights
    #autoencoder.layers[0].set_weights([q,w,e,r])
    autoencoder.layers[1].set_weights([x,y,z])#,w])  #z DNE before



#x = autoencoder.layers[1].weights
#print(x)


#optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=5e-6)#was -7//// #5e-3)



    input_size=2048
    idx = 2048
    rng = np.random.default_rng()
#read data
    x = np.fromfile("../data/dataset/lfm_test_10M_100m_0000.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)
    #x = np.fromfile("../data/dataset/VH1-164.sigmf-data.bin", dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32)
    x = x[np.s_[idx:idx+input_size]]
    
    #scale because cant have negative numbers
    x = x +4096

    X = tf.convert_to_tensor(x)
    X = tf.reshape(X, [-1, 2048])





#x = rng.integers(0, 4095, size=(1, 1, 1, 16), dtype=np.int16, endpoint=False).astype(np.float32)
#X=tf.convert_to_tensor(x)
#X=tf.reshape(X, [-1,16])
#run the model



#autoencoder.compile(optimizer=opt, loss=losses.CategoricalHinge())
    autoencoder.compile(optimizer=opt, loss=losses.MeanSquaredError())
    history = autoencoder.fit(X,X, epochs=20000000)


    numpy_loss = history.history["loss"]
    #numpy_y = numpy_loss[399990]
    #numpy_x = np.arange(1)
    #plt.plot(numpy_y, numpy_x)
    #plt.scatter()
    
    
    #y_plot_secret.append(numpy_loss[399990])
    #y_plot_secret2.append(numpy_loss[390000])
    #if(numpy_loss[399990] > 10000):
    #    numpy_loss[399990] = 10500
    #if(numpy_loss[399990] < 600):
    #    output_array.append(autoencoder.predict(X))
    #x_plot.append(numpy_loss[399990])
    #y_plot.append(counter)
    
    #following lines are for plotting the testing of layers
    
    x_plot = numpy_loss
    y_plot = np.arange(150000)

    #plt.plot(y_plot, x_plot)

    counter = counter +1


    if(printing):
        f = open('./output_weights_v2.txt', 'a+')
        print("*************************************************************************************************************************************", file = f)
        print(autoencoder.layers[0].weights, file =f)
        print("--------",file = f)
        f.close()



#print(tf.shape(autoencoder.layers[1].get_weights()))


#print(X)

predictions = autoencoder.predict(X)
#print(predictions)
differnce = X - predictions
print(differnce)

#################################3
print("#########################")
np.set_printoptions(threshold=np.inf)
#print(autoencoder.layers[0].weights)
print("#########################")

#f = open('./weights_noreluall_4096_2048.txt', 'a')
#print(autoencoder.layers[1].weights, file = f)
#f.close()


#print(history.history["loss"])
#numpy_loss = np.array(history.history["loss"])
#print(numpy_loss[390000])
#numpy_arr = np.arange(400000)


#import matplotlib.pyplot as plt
#plt.plot(numpy_arr, numpy_loss)

if(graphing):
    #print(y_plot, x_plot)
    #plt.scatter(y_plot,x_plot)
    #plt.plot(y_plot, x_plot)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(["Dataset 1", "Dataset 2","Dataset 3","Dataset 4"])
    plt.savefig("testing_index2048-4096-frozendecoder_presaved-weights.jpg")

#print(X)
#print(output_array)

#print(y_plot_secret)
#print(y_plot_secret2)


