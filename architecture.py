import tensorflow as tf
import sys

from tf_ops import *

'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2, name='lrelu'):
   return tf.maximum(leak*x, x)

def netG(z, batch_size):
   print 'GENERATOR'
   z = tf.layers.dense(z, 4*4*1024, name='g_z')
   z = tf.reshape(z, [batch_size, 4, 4, 1024])
   z = relu(bn(z))
   
   conv1 = tf.layers.conv2d_transpose(z, 512, 5, strides=2, padding='SAME', name='g_conv1')
   conv1 = relu(bn(conv1))

   conv2 = tf.layers.conv2d_transpose(conv1, 256, 5, strides=2, padding='SAME', name='g_conv2')
   conv2 = relu(bn(conv2))

   conv3 = tf.layers.conv2d_transpose(conv2, 128, 5, strides=2, padding='SAME', name='g_conv3')
   conv3 = relu(bn(conv3))

   conv4 = tf.layers.conv2d_transpose(conv3, 3, 5, strides=2, padding='SAME', name='g_conv4')
   conv4 = tanh(conv4)
   
   print 'z:',z
   print 'conv1:',conv1
   print 'conv2:',conv2
   print 'conv3:',conv3
   print 'conv4:',conv4
   print
   print 'END G'
   print
   tf.add_to_collection('vars', z)
   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)

   return conv4 


'''
   Discriminator network
'''
def netD(input_images, BATCH_SIZE, reuse=False):
   print 'DISCRIMINATOR reuse = '+str(reuse)
   
   conv1 = tf.layers.conv2d(input_images, 64, 5, strides=2, reuse=reuse, padding='SAME', name='d_conv1')
   conv1 = lrelu(conv1)

   conv2 = tf.layers.conv2d(conv1, 128, 5, strides=2, reuse=reuse, padding='SAME', name='d_conv2')
   conv2 = lrelu(conv2)

   conv3 = tf.layers.conv2d(conv2, 256, 5, strides=2, reuse=reuse, padding='SAME', name='d_conv3')
   conv3 = lrelu(conv3)

   conv4 = tf.layers.conv2d(conv3, 512, 5, strides=2, reuse=reuse, padding='SAME', name='d_conv4')
   conv4 = lrelu(conv4)

   conv4_flat = tf.reshape(conv4, [BATCH_SIZE, -1])
   dense = tf.layers.dense(conv4_flat, 1, reuse=reuse, name='d_dense')
   return dense

   #conv5 = tf.layers.conv2d(conv4, 1, 4, strides=1, reuse=reuse, padding='SAME', name='d_conv5')
   #return conv5

   conv5_flat = tf.reshape(conv5, [BATCH_SIZE, -1])
   dense = lrelu(tf.layers.dense(conv5_flat, 1, reuse=reuse, name='d_dense'))

   print 'input images:',input_images
   print 'conv1:',conv1
   print 'conv2:',conv2
   print 'conv3:',conv3
   print 'conv4:',conv4
   print 'conv5:',conv5
   print 'dense:',dense
   print

   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)
   tf.add_to_collection('vars', conv5)
   tf.add_to_collection('vars', dense)
   return dense

