import tensorflow as tf

class Conv2D_quant(tf.keras.layers.Layer):
    def __init__(self,th,filters=16,kernel_size=(3, 3),padding="valid",activation='relu',input_shape = None):
        super(Conv2D_quant, self).__init__()
        self.filters = filters
        self.kernel_size=kernel_size
        self.padding=padding
        self.activation=activation
        self.input_shape1 = input_shape
        self.th = th
    
    def build(self,input_shape):
      if self.input_shape1:
        self.linear_1 = tf.keras.layers.Conv2D(filters=self.filters,kernel_size=self.kernel_size,padding=self.padding,input_shape = self.input_shape1)
      else:
        self.linear_1 = tf.keras.layers.Conv2D(filters=self.filters,kernel_size=self.kernel_size,padding=self.padding,input_shape = [int(input_shape[-1])])
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
    def __init__(self, th, num_output, activation = 'relu', input_shape = None):
        super(Dense_quant, self).__init__()
        self.num_output = num_output
        self.activation=activation
        self.input_shape1 = input_shape
        self.th = th
    
    def build(self,input_shape):
      if self.input_shape1:
        self.linear_1 = tf.keras.layers.Dense(self.num_output,input_shape = self.input_shape1)
      else:
        self.linear_1 = tf.keras.layers.Dense(self.num_output,input_shape = [int(input_shape[-1])])
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