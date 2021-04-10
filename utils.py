import numpy as np
import tensorflow as tf

s = tf.math.sign

#what does this do? 
def truncate(grad,bit_width):
  return tf.floor(grad*(2.0**bit_width))/(2.0**bit_width)
 
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

def update_fast_new(grad,trainable_weights,lr, refresh_cycle, lsb_width, msb_width):
  
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
    max_lsb = 2**(-msb_width) - 2**(-total_width)
    weight_lsb = tf.clip_by_value(weight_lsb,0,max_lsb)
 
    refresh_term = (2**(-total_width))*(refresh_cycle/2.0)*( tf.math.sign(weight_lsb-2**(-total_width-5)) + tf.math.sign(weight_lsb-(max_lsb-2**(-total_width-5))))
 
    answer[i] =  -(tf.convert_to_tensor(weight_msb + weight_lsb+refresh_term)-trainable_weights[i])
 
  return answer
 
# @tf.function
def update_fast(grad,trainable_weights,lr):
  answer = (grad.copy())
  
  for i in range(len(grad)):
      total_width = 11
      # print(i)
      # # answer[i] = 0.0005*(0.5*(tf.math.sign(grad[i] - 0.1)+tf.math.sign(grad[i]+0.1)))
      final_weights = trainable_weights[i]-lr*grad[i]
      # final_weights*= 0.988
      # tf.print(tf.math.reduce_mean(tf.math.abs(grad[i])))
      final_weights = final_weights*(tf.math.sign(tf.math.abs(final_weights)-(2**(-total_width)-2**(-total_width-5)))+1)/2
      final_weights = tf.clip_by_value(final_weights,clip_value_min=-1, clip_value_max=1)
      answer[i] = -(quantize(final_weights,total_width)-trainable_weights[i])
      # answer[i] = grad[i]*lr
      # answer[i] = answer[i]*(tf.math.sign(tf.math.abs(answer[i])-(2**(-total_width)-2**(-total_width-5)))+1)/2
      # answer[i] = quantize(answer[i],total_width)
 
  return answer
  