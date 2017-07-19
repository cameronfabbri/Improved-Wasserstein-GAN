'''

   Operations commonly used in tensorflow

'''

import tensorflow as tf
import numpy as np
import math


'''
   Batch normalization
   https://arxiv.org/abs/1502.03167
'''
def bn(x):
   return tf.layers.batch_normalization(x)

'''
   Instance normalization
   https://arxiv.org/abs/1607.08022
'''
def instance_norm(x, epsilon=1e-5):
   mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
   return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

'''
   2d transpose convolution, but resizing first then performing conv2d
   with kernel size 1 and stride of 1
   See http://distill.pub/2016/deconv-checkerboard/

   The new height and width can be anything, but default to the current shape * 2
'''
def upconv2d(x, filters, name=None, new_height=None, new_width=None, kernel_size=3):

   print 'x:',x
   shapes = x.get_shape().as_list()
   height = shapes[1]
   width  = shapes[2]

   # resize image using method of nearest neighbor
   if new_height is None and new_width is None:
      x_resize = tf.image.resize_nearest_neighbor(x, [height*2, width*2])
   else:
      x_resize = tf.image.resize_nearest_neighbor(x, [new_height, new_width])

   # conv with stride 1
   return tf.layers.conv2d(x_resize, filters, kernel_size, strides=1, name=name)


######## activation functions ###########
'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2):
   return tf.maximum(leak*x, x)

'''
   Like concatenated relu, but with elu
   http://arxiv.org/abs/1603.05201
'''
def concat_elu(x):
   axis = len(x.get_shape())-1
   return tf.nn.elu(tf.concat(values=[x, -x], axis=axis))

'''
   Concatenated ReLU
   http://arxiv.org/abs/1603.05201
'''
def concat_relu(x):
   axis = len(x.get_shape())-1
   return tf.nn.relu(tf.concat([x, -x], axis))

'''
   Regular relu
'''
def relu(x, name='relu'):
   return tf.nn.relu(x)

'''
   Tanh
'''
def tanh(x):
   return tf.nn.tanh(x)

'''
   Sigmoid
'''
def sig(x):
   return tf.nn.sigmoid(x)

'''
   Self normalizing neural networks paper
   https://arxiv.org/pdf/1706.02515.pdf
'''
def selu(x):
   alpha = 1.6732632423543772848170429916717
   scale = 1.0507009873554804934193349852946
   return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

'''
   Like concat relu/elu, but with selu
'''
def concat_selu(x):
   axis = len(x.get_shape())-1
   return selu(tf.concat([x, -x], axis))

###### end activation functions #########





