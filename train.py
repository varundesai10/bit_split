import tensorflow as tf
from utils import *
from layers import Conv2D_quant, Dense_quant
from model_builder import build_model
import ssl
from get_dataset import *
import argparse

#tf version 2.4.0 needed to run. won't run on other versions of tf.
#example: python3 train.py -r 10 -W -w 0.06 -D -d 0.04 -t 15 -l 6 -L -S "mnist"

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--refresh', help = "Refresh Cycle Frequency", type = int, default = 10)
parser.add_argument('-W', '--write_noise', help = "Add write noise", action = 'store_true')
parser.add_argument('-w', '--write_std_dev', help = "Write noise std dev", type = float, default = 0.06)
parser.add_argument('-D', '--device_variation', help = "Add device variations", action = 'store_true')
parser.add_argument('-d', '--device_std_dev', help = "Device Var Std Dev", type = float, default = 0.04)
parser.add_argument('-t', '--total_precision', help = "Total Bit Width", type = int, default = 15)
parser.add_argument('-l', '--lsb_precision', help = "LSB_precision", type = int, default = 6)
parser.add_argument('-L', '--load_prev', help = "Load Previous Weight values", action = 'store_true')
parser.add_argument('-P', '--prefix', help = "Prefix to where the results are stored", type = str, default = "")
parser.add_argument('-S', '--dataset', help = "which dataset to use", type = str, default = "cifar10")
args = parser.parse_args()

#fixing errors which occur on HPC
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


model = build_model(th = 0.1, dv = args.device_variation, std_dev = args.device_std_dev, add_zero_pad = (not args.dataset == "cifar10"))
opt = tf.keras.optimizers.SGD(learning_rate=1)
model.compile(optimizer=opt,
             loss='categorical_crossentropy',
             metrics=['accuracy'])

if("mnist" in args.dataset):
    model.build(input_shape = tf.TensorShape([None, 28, 28, 1]))
else:
    model.build(input_shape=tf.TensorShape([None,32, 32, 3]))
print(model.summary())
dataset, dataset_test = build_dataset(args.dataset)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
base_path = os.path.join(os.path.abspath("./results/"), args.prefix, "res_{}_{}_{}_{}_{}_{}_{}".format(args.refresh,
                                                                                        args.total_precision, 
                                                                                        args.lsb_precision, 
                                                                                        args.write_noise, 
                                                                                        args.write_std_dev, 
                                                                                        args.device_variation, 
                                                                                        args.device_std_dev))
s = "Running with the Following Parameters:\nRefresh Cycle: {} Total Precision: {} LSB Precision: {}\
    Write Noise: {} STD: {} DeviceVar: {} STD: {}\
    Load Prev: {} Prefix: {} Base Path: {}".format(args.refresh,
        args.total_precision, 
        args.lsb_precision, 
        args.write_noise, 
        args.write_std_dev, 
        args.device_variation, 
        args.device_std_dev, 
        args.load_prev, args.prefix, base_path)
print(s)

#training:
acc_hist = fast_backprop(dataset = dataset,
                        dataset_test = dataset_test,
                        epochs = 50,
                        model = model,
                        loss_fn = loss_fn,
                        opt = opt,
                        msb = args.total_precision - args.lsb_precision,
                        lsb = args.lsb_precision,
                        write_noise = args.write_noise, 
                        std_dev = args.write_std_dev,
                        refresh_freq = args.refresh, 
                        load_prev_val = args.load_prev, 
                        base_path = base_path)





