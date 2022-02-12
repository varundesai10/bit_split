import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.layers.ops import core as core_ops

	
class Conv2D_quant(tf.keras.layers.Layer):
	def __init__(self,th = 0.1,filters=16,
				kernel_size=(3, 3),padding="valid",
				activation='relu',input_shape = None,
				dv = False, std_dev = 0.06):
		super(Conv2D_quant, self).__init__()
		self.filters = filters
		self.kernel_size=kernel_size
		self.padding=padding
		self.activation=activation
		self.input_shape1 = input_shape
		self.th = th
		self.dv = dv
		self.std_dev = 0.06
    
	def build(self,input_shape):
		if self.dv:
			if self.input_shape1:
				self.linear_1 = Conv2D_dv(filters=self.filters,kernel_size=self.kernel_size, std_dev = self.std_dev, padding=self.padding,input_shape = self.input_shape1)
			else:
				self.linear_1 = Conv2D_dv(filters=self.filters,kernel_size=self.kernel_size, std_dev = self.std_dev, padding=self.padding,input_shape = [int(input_shape[-1])])
		
		else:
			i_s = self.input_shape1 if self.input_shape1 is not None else [int(input_shape[-1])]
			self.linear_1 = tf.keras.layers.Conv2D(filters=self.filters,kernel_size=self.kernel_size, padding=self.padding,input_shape =i_s)
		self.activation = tf.keras.layers.Activation(self.activation)
 
	@tf.custom_gradient
	def call(self, x):        
		with tf.GradientTape() as tape1:
			tape1.watch(x)
			th = self.th
			x_th = x+tf.stop_gradient(0.5*(tf.math.sign(x-th)+tf.math.sign(x+th))-x)
			result = self.linear_1(x_th)+tf.stop_gradient(self.linear_1(x)-self.linear_1(x_th))
			y = self.activation(result)
			
			def backward(dy, variables = None):
				grad = tape1.gradient(y,[x_th]+self.trainable_weights, output_gradients = dy)
				return (grad[0], grad[1:])
		return y,backward

class Dense_quant(tf.keras.layers.Layer):
	def __init__(self, th, num_output, activation = 'relu', input_shape = None, dv = False, std_dev = 0.06):
		super(Dense_quant, self).__init__()
		self.num_output = num_output
		self.activation=activation
		self.input_shape1 = input_shape
		self.th = th
		self.dv = dv
		self.std_dev = std_dev
    
	def build(self,input_shape):
		if not self.dv:
			if self.input_shape1:
				self.linear_1 = tf.keras.layers.Dense(self.num_output,input_shape = self.input_shape1)
			else:
				self.linear_1 = tf.keras.layers.Dense(self.num_output, input_shape = [int(input_shape[-1])])
		
		else:
			if self.input_shape1:
				self.linear_1 = Dense_dv(self.num_output,input_shape = self.input_shape1, std_dev = self.std_dev)
			else:
				self.linear_1 = Dense_dv(self.num_output, input_shape = [int(input_shape[-1])], std_dev = self.std_dev)
		self.activation = tf.keras.layers.Activation(self.activation)
    
	@tf.custom_gradient
	def call(self, x):        
		with tf.GradientTape() as tape1:
			tape1.watch(x)
			th = self.th
			x_th = x+tf.stop_gradient(0.5*(tf.math.sign(x-th)+tf.math.sign(x+th))-x)
			result = self.linear_1(x_th)+tf.stop_gradient(self.linear_1(x)-self.linear_1(x_th))
			y = self.activation(result)
			
			def backward(dy, variables = None):
				grad = tape1.gradient(y,[x_th]+self.trainable_weights, output_gradients = dy)
				return (grad[0], grad[1:])
		return y,backward

class Conv2D_dv(tf.keras.layers.Conv2D):
	def __init__(self,
               filters,
               kernel_size,
			   std_dev = 0.06,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
		super(Conv2D_dv, self).__init__(filters,
               kernel_size,
               strides,
               padding,
               data_format,
               dilation_rate,
               groups,
               activation,
               use_bias,
               kernel_initializer,
               bias_initializer,
               kernel_regularizer,
               bias_regularizer,
               activity_regularizer,
               kernel_constraint,
               bias_constraint,
               **kwargs)
		self.std_dev = std_dev

	def build(self, input_shape):
		super().build(input_shape)
		self.kernel_dv = tf.constant(tf.random.normal(self.kernel.shape, 1, self.std_dev))
		self.bias_dv = tf.constant(tf.random.normal(self.bias.shape, 1, self.std_dev))
	
	def call(self, inputs):
		if self._is_causal:  # Apply causal padding to inputs for Conv1D.
			inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

		kernel_ = self.kernel + tf.stop_gradient(tf.multiply(self.kernel, self.kernel_dv) - self.kernel)
		bias_ = self.bias + tf.stop_gradient(tf.multiply(self.bias, self.bias) - self.bias)
		outputs = self._convolution_op(inputs, kernel_)

		if self.use_bias:
			output_rank = outputs.shape.rank
			if self.rank == 1 and self._channels_first:
				# nn.bias_add does not accept a 1D input tensor.
				bias = array_ops.reshape(bias_, (1, self.filters, 1))
				outputs += bias
			else:
				# Handle multiple batch dimensions.
				if output_rank is not None and output_rank > 2 + self.rank:
					def _apply_fn(o):
						return nn.bias_add(o, bias_, data_format=self._tf_data_format)

					outputs = nn_ops.squeeze_batch_dims(
              			outputs, _apply_fn, inner_rank=self.rank + 1)
				else:
					outputs = nn.bias_add(
              		outputs, bias_, data_format=self._tf_data_format)

		if self.activation is not None:
			return self.activation(outputs)
    	
		return outputs 

class Dense_dv(tf.keras.layers.Dense):
	def __init__(self,
               units,
			   std_dev = 0.06,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
		super(Dense_dv, self).__init__(units,
               activation,
               use_bias,
               kernel_initializer,
               bias_initializer,
               kernel_regularizer,
               bias_regularizer,
               activity_regularizer,
               kernel_constraint,
               bias_constraint,
               **kwargs)
		
		self.std_dev = std_dev
	
	def build(self, input_shape):
		super().build(input_shape)
		self.kernel_dv = tf.constant(tf.random.normal(self.kernel.shape, 1, self.std_dev))
		self.bias_dv = tf.constant(tf.random.normal(self.bias.shape, 1, self.std_dev))
	
	def call(self, inputs):
		kernel = self.kernel + tf.stop_gradient(tf.math.multiply(self.kernel, self.kernel_dv) - self.kernel)
		bias = self.bias + tf.stop_gradient(tf.math.multiply(self.bias, self.bias_dv) - self.bias)
		return core_ops.dense(
        inputs,
        kernel,
        bias,
        self.activation,
        dtype=self._compute_dtype_object)
