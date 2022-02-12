import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def build_dataset():
    (x_train, y_train_), (x_test, y_test_) = cifar10.load_data()
    x_train = 2*x_train.astype('float32') / 255 -1
    x_test = 2*x_test.astype('float32') / 255 -1

    y_train = to_categorical(y_train_)
    y_test = to_categorical(y_test_)

    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train_)
    )
    dataset = dataset.shuffle(buffer_size=1024).batch(1)
    dataset_test = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test_)
    )
    dataset_test = dataset_test.shuffle(buffer_size=1024).batch(1)

    return dataset, dataset_test