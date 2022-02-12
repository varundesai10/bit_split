import tensorflow as tf
from utils import *
from layers import Conv2D_quant, Dense_quant
from model_builder import build_model
import ssl
from get_dataset import *
import argparse

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

 
model = build_model()
opt = tf.keras.optimizers.SGD(learning_rate=1)
model.compile(optimizer=opt,
             loss='categorical_crossentropy',
             metrics=['accuracy'])
model.build(input_shape=tf.TensorShape([None,32, 32, 3]))
print(model.summary())
dataset, dataset_test = build_dataset()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

loss = 0
acc = 0

acc_hist = fast_backprop(dataset, dataset_test, 50, model, loss_fn, opt, 9, 6, True, 0.05, 10, False, './')





