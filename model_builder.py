import tensorflow as tf
from tensorflow.keras.layers import Dropout, MaxPool2D, Flatten, ZeroPadding2D
from layers import Dense_quant, Conv2D_quant

def build_model(th = 0.1, shape = 1, act = 'selu', write_noise = False, write_noise_std_dev = 0.02, device_var = False, device_var_std_dev = 0.06):
    filter_shape = (2*shape+1,2*shape+1)
    model = tf.keras.models.Sequential()
    model.add(Dropout(.4))
    model.add(ZeroPadding2D(padding=shape))
    model.add(Conv2D_quant(th,filters=64, 
                    kernel_size=filter_shape,
                    activation=act,
                    input_shape=(32, 32, 3)))
    model.add(Dropout(.4))
    # model.add(MaxPool2D())
    model.add(ZeroPadding2D(padding=shape))
    model.add(Conv2D_quant(th,filters=64,
                    kernel_size=filter_shape,
                    activation=act))
    model.add(MaxPool2D())
    model.add(Dropout(.4))
    model.add(ZeroPadding2D(padding=shape))
    model.add(Conv2D_quant(th,filters=64, 
                    kernel_size=(3, 3),
                    activation=act))
                    # input_shape=(32, 32, 3)))
    # model.add(MaxPool2D())
    model.add(Dropout(.4))
    model.add(ZeroPadding2D(padding=shape))
    model.add(Conv2D_quant(th,filters=64,
                    kernel_size=(3, 3),
                    activation=act))
    model.add(MaxPool2D())
    model.add(Dropout(.4))
    model.add(ZeroPadding2D(padding=shape))
    model.add(Conv2D_quant(th,filters=64,
                    kernel_size=filter_shape,
                    activation=act))
    # model.add(MaxPool2D())
    model.add(Dropout(.4))
    model.add(ZeroPadding2D(padding=shape))
    model.add(Conv2D_quant(th,filters=64,
                    kernel_size=filter_shape,
                    activation=act))
    model.add(MaxPool2D())
    model.add(Dropout(.4))
    model.add(ZeroPadding2D(padding=shape))
    model.add(Conv2D_quant(th,filters=128,
                    kernel_size=filter_shape,
                    activation=act))
    model.add(MaxPool2D())
    # model.add(Dropout(.4))
    # model.add(ZeroPadding2D(padding=shape))
    # model.add(Conv2D_quant(th,filters=128,
    #                 kernel_size=filter_shape,
    #                 activation=act))
    # model.add(MaxPool2D())
    model.add(Dropout(.4))
    model.add(Flatten())
    #model.add(Dense(100, activation = act))
    # model.add(Dropout(.4))
    model.add(Dense_quant(th,100, activation = act))
    # model.add(Dropout(.4))
    model.add(Dense_quant(th,10, activation='softmax'))

    return model