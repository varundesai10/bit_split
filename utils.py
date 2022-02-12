import numpy as np
import tensorflow as tf
import time
import os
import pandas as pd

s = tf.math.sign

#what does this do? 
def truncate(grad,bit_width):
	return tf.math.round(grad*(2.0**bit_width))/(2.0**bit_width)
 
# @tf.function
def quantize(grad,bit_width):
	sign_p = tf.math.sign(grad)
	temp = tf.math.abs(grad)
	answer = tf.floor(temp*(2.0**bit_width))/(2.0**bit_width)
	return tf.math.multiply(answer, sign_p)

def weight_decay(w, msb_width = 9, lsb_width = 5, is_lsb = True):
	#y = Ae^-bx + y_0
  total_width = msb_width + lsb_width
  
  answer = w.copy()
  A = tf.constant([1.89208984e-03,8.47417971e-04,5.94509440e-04,4.39956507e-04,3.30262930e-04,2.46271971e-04,1.78911805e-04,1.23176272e-04,7.60085713e-05,3.54098283e-05,0])
  alpha = tf.constant([0.92935259, 0.96917519, 0.97447399, 0.97645794, 0.97690667, 0.97626783, 0.97453573, 0.9713371 ,0.96564474,0.95452046])
  a = tf.constant(486.4834783989868)
  b = tf.constant(0.7273794803789448)
  
  for i in range(len(answer)):
    
    answer[i] = tf.zeros_like(w[i])
    y_0 = 0.7
    w_msb = truncate(w[i],msb_width)
    w_lsb = truncate(w[i]-w_msb,total_width)
    
    for j in range(len(A) - 1):
      
      temp = 0.5*(s(w_lsb - A[j+1]) - s(w_lsb - A[j])) * (alpha[j]*(w_lsb) + (b/a)*(alpha[j] - 1) + (y_0/a)*(b-y_0))
      answer[i] += temp

    answer[i] = tf.abs(answer[i])
    answer[i] = -((w_msb + answer[i]) - w[i])

  #return -((w_msb + answer) - w), tf.reduce_mean(tf.abs(w - answer))
  return answer 

def update_fast(grad,trainable_weights,lr, refresh_cycle, lsb_width, msb_width, write_noise, std_dev):
	answer = (grad.copy())
	for i in range(len(grad)):
		curr_weights = tf.clip_by_value(trainable_weights[i],clip_value_min=-1, clip_value_max=1)
		total_width = msb_width+lsb_width
		curr_weights = curr_weights * ( tf.math.sign ( tf.math.abs(curr_weights) - (2**(-total_width)-2**(-total_width-5)) ) +1 )/2
		update_w = lr*grad[i]
		weight_msb = truncate(curr_weights,msb_width)
		weight_lsb = truncate(curr_weights-weight_msb,total_width)
		update_w = tf.clip_by_value(update_w,-2**(-msb_width),2**(-msb_width))

		update_w = quantize(update_w, total_width)

		weight_lsb = weight_lsb - update_w
		# max_lsb = 2**(-msb_width) - 2**(-total_width)
		max_lsb = 2**(-msb_width-1)
		weight_lsb = tf.clip_by_value(weight_lsb,-max_lsb,max_lsb)
		refresh_term = (2**(-total_width))*(refresh_cycle/2.0)*( tf.math.sign(weight_lsb-(-max_lsb + 2**(-total_width-5))) + tf.math.sign(weight_lsb-(max_lsb-2**(-total_width-5))))
		#if the w_lsb is close to -max_lsb or +max_lsb then we update it to a lower or higher value.
		answer[i] =  -(tf.convert_to_tensor(weight_msb + weight_lsb+refresh_term)-trainable_weights[i])
		#if(write_noise):
		#	answer[i] = tf.math.multiply(answer[i], tf.random.normal(answer[i].shape, 1, std_dev))
	return answer

@tf.function      
def fast_backprop_single_sample(x,y,model,loss_fn,opt,temp,acc,lr, msb, lsb, refresh_cycle,write_noise=False, std_dev=0.02): 
	with tf.GradientTape() as tape:
		logits = model(x) #Forward Pass
		loss = loss_fn(y, logits) #Getting Loss
		temp += loss
		y_int64 = (tf.cast(y,tf.int64))
		pred = (tf.equal(tf.math.argmax(logits,1),y_int64))
		acc = tf.cond(pred, lambda: tf.add(acc,1), lambda: tf.add(acc,0))
		gradients = tape.gradient(loss, model.trainable_weights) #get gradients
		gradients = update_fast(gradients,model.trainable_weights,lr, refresh_cycle, lsb, msb, write_noise, std_dev)
		opt.apply_gradients(zip(gradients, model.trainable_weights))
	return temp,acc

def fast_backprop(dataset = None, dataset_test = None, 
					epochs = 50, model = None, loss_fn = None, opt = None, msb = 9, lsb = 6, write_noise = False, std_dev = 0.02, refresh_freq = 10, load_prev_val = False, base_path = './'):
	acc_hist = []
	test_acc = []
	base_lr = 1e-3
	weight_suffix = 'weights'
	accuracy_suffix = 'acc'
	if(load_prev_val):
		acc_hist = list(pd.read_csv(os.path.join(base_path, 'acc', 'training_acc.csv')).to_numpy()[:,1])
		test_acc = list(pd.read_csv(os.path.join(base_path, 'acc', 'test_acc.csv')).to_numpy()[:,1])
		base_lr = pd.read_csv(os.path.join(base_path, 'acc', 'learning_rate.csv')).to_numpy()[0,1]
		tf.print("Base lr =", base_lr, "acc_hist = ", acc_hist, "test_acc = ", test_acc);
  
	for epoch in range(epochs):
		print("*"*25, "Epoch {}".format(epoch), "*"*25, sep='')	
		tm = time.time()
		acc = 0.0
		temp = 0.0
		iterator = iter(dataset)
		lr = base_lr*(0.985**(tf.math.floor(epoch/1.0)))
		print("Learning rate = ", lr.numpy())
		
		for step in range(len(dataset)):
			x,y = iterator.get_next()
			refresh_cycle = 1 if step % refresh_freq == 0 else 0;
			temp,acc = fast_backprop_single_sample(x,y,model,loss_fn,opt,temp,acc,lr,msb, lsb, refresh_cycle, write_noise, std_dev)

			if step%1000 == 999: #printing stuff
				tf.print("Time taken: ",time.time()-tm)
				tm = time.time()
				step_float = tf.cast(step,tf.float32)
				tf.print("Step:", step, "Loss:", float(temp/step_float))
				tf.print("Train Accuracy: ",acc*100.0/step_float)
				#model.save_weights(os.path.join(base_path, weight_suffix, 'weights.h5'))
				tf.print("Weights saved!")
				acc_hist.append(acc.numpy()*100.0/step_float.numpy())
				#pd.DataFrame({'acc':acc_hist}).to_csv(os.path.join(base_path, accuracy_suffix, 'training_acc.csv'))

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
					test_acc.append(acc_test.numpy()*100/tf.cast(step_test, tf.float32).numpy())
					pd.DataFrame({'acc_test':test_acc}).to_csv(os.path.join(base_path, accuracy_suffix, 'test_acc.csv'))
		
		step_float = tf.cast(step,tf.float32)
		pd.DataFrame({'lr': list( [lr.numpy()] ) }).to_csv(os.path.join(base_path, accuracy_suffix, 'learning_rate.csv'))
  
	return acc_hist