from layers import InputThreshold, Dense_quant
import tensorflow as tf

m1 = tf.keras.models.Sequential([InputThreshold(0.1), tf.keras.layers.Dense(100, activation='relu')]);
m2 = tf.keras.models.Sequential([Dense_quant(0.1, 100)])

m1.build(input_shape = (None, 200))
m2.build(input_shpe = (None, 200))
