from utils import update_fast_new, weight_decay
from layers import Conv2D_quant, Dense_quant

import tensorflow as tf
from tensorflow import keras
layers = keras.layers
cifar10 = keras.datasets.cifar10
ExponentialDecay = keras.optimizers.schedules.ExponentialDecay
BatchNormalization = tf.keras.layers.BatchNormalization
regularizers = keras.regularizers
import time
to_categorical = keras.utils.to_categorical
Sequential = keras.models.Sequential
Dropout = layers.Dropout
MaxPool2D = layers.MaxPool2D
Flatten = layers.Flatten
'''
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam, SGD
from keras.optimizers.schedules import ExponentialDecay
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers import Dropout
import tensorflow as tf
import time
'''

(x_train, y_train_), (x_test, y_test_) = cifar10.load_data()
x_train = 2*x_train.astype('float32') / 255 -1  #limits the input from -1 to +1
x_test = 2*x_test.astype('float32') / 255 -1    


y_train = to_categorical(y_train_)   #converts the type of the data from N_samples x 1 to N_samples x N_classes
y_test = to_categorical(y_test_)
 

model = Sequential()

th = 0.1 #thresholding of input
act = 'selu'

model.add(Conv2D_quant(th,filters=64, 
                kernel_size=(3, 3),
                padding="same",
                activation=act,
                input_shape=(32, 32, 3)))

model.add(Dropout(.4))

model.add(Conv2D_quant(th,filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation=act))

model.add(MaxPool2D())

model.add(Dropout(.4))

model.add(Conv2D_quant(th,filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation=act))

model.add(Dropout(.4))

model.add(Conv2D_quant(th,filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation=act))

model.add(MaxPool2D())

model.add(Dropout(.4))

model.add(Conv2D_quant(th,filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation=act))

model.add(Dropout(.4))

model.add(Conv2D_quant(th,filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation=act))

model.add(MaxPool2D())

model.add(Dropout(.4))

model.add(Conv2D_quant(th,filters=32,
                kernel_size=(3, 3),
                padding="same",
                activation=act))

model.add(Flatten())

model.add(Dropout(.4))

model.add(Dense_quant(th,10, activation='softmax'))

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.002,decay_steps=50000,decay_rate=0.96,staircase=True)

opt = keras.optimizers.SGD(learning_rate=1)

model.compile(optimizer=opt,
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.build(input_shape=(None,32, 32, 3))

print(model.summary())

dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train_)
)
#creates a dataset class from x_train, y_train


dataset = dataset.shuffle(buffer_size=1024).batch(1) #shuffles the dataset, sets batch size to 1, to mimic as what is done in the circuit.

dataset_test = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test_)
)

dataset_test = dataset_test.shuffle(buffer_size=1024).batch(1)

x_train = tf.convert_to_tensor(x_train)
y_train_ = tf.convert_to_tensor(y_train_)
x_test = tf.convert_to_tensor(x_test)
y_test_ = tf.convert_to_tensor(y_test_)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
loss = 0
acc = 0

def fast_backprop_single_sample(x,y,model,loss_fn,opt,temp,acc,lr,refresh_cycle):
      with tf.GradientTape() as tape:
        # Forward pass.
        logits = model(x)
 
        # External loss value for this batch.
        loss = loss_fn(y, logits)
        # tf.print(loss)        
        temp+= loss
 
        y_int64 = (tf.cast(y,tf.int64))
        pred = (tf.equal(tf.math.argmax(logits,1),y_int64))
        acc = tf.cond(pred, lambda: tf.add(acc,1), lambda: tf.add(acc,0))
 
        # Add the losses created during the forward pass.
        loss += sum(model.losses)
 
        
        
        # gradients = update_fast(gradients,model.trainable_weights,lr)
        total_bits = 15-1
        lsb = 5
        
        # Get gradients of weights wrt the loss.
        gradients = tape.gradient(loss, model.trainable_weights)
        gradients = update_fast_new(gradients,model.trainable_weights,lr, refresh_cycle, lsb, total_bits-lsb)
        opt.apply_gradients(zip(gradients, model.trainable_weights))

        # get decay values!
        decay = weight_decay(model.trainable_weights)
        #apply decay
        opt.apply_gradients(zip(decay, model.trainable_weights))
        # tape.reset()
    
        # Logging.
 
      return temp,acc
 
def fast_backprop(dataset,epochs,model,loss_fn,opt):
  tm = 0.0
  for epoch in range(epochs):
    # dataset = dataset.shuffle(buffer_size=1024)
 
    # print(tm-time.time())
    tm = time.time()
    acc = 0.0
    temp = 0.0
    # step = 0
    iterator = iter(dataset)
    lr = 0.001*(0.97**(tf.math.floor(epoch/1.0)))
    # print(iterator)
    for step in range(len(dataset)):#(x, y) in (dataset):
      x,y = iterator.get_next()
      if step%100 == 99:
        refresh_cycle = 1
      else:
        refresh_cycle = 0
 
      temp,acc = fast_backprop_single_sample(x,y,model,loss_fn,opt,temp,acc,lr, refresh_cycle)
      if step%10000 == 9999:
          tf.print("Time taken: ",time.time()-tm)
          tm = time.time()
          step_float = tf.cast(step,tf.float32)
          tf.print("Step:", step, "Loss:", float(temp/step_float))
          tf.print("Train Accuracy: ",acc*100.0/step_float)
      if step % 50000 == 49999:
          step_test = 0
          acc_test = 0.0
          for x_test_i, y_test_i in dataset_test:
            step_test+=1
            logits = model(x_test_i)
            y_int64 = (tf.cast(y_test_i,tf.int64))
            pred = (tf.equal(tf.math.argmax(logits,1),y_int64))
            acc_test = tf.cond(pred, lambda: tf.add(acc_test,1), lambda: tf.add(acc_test,0))
          tf.print("Test Accuracy: ", acc_test*100/tf.cast(step_test,tf.float32))
      step_float = tf.cast(step,tf.float32)

# time_time = time.time()
fast_backprop(dataset,100,model,loss_fn,opt)
# print(time.time()-time_time)
# history = model.fit(x_train, y_train, batch_size=1, epochs=1000, verbose=1, validation_data=(x_test, y_test))





