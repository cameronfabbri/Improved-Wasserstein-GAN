import tensorflow as tf
import tensorflow.contrib.layers as tcl
import sys
from tf_ops import *

def netG(z, batch_size):
   print 'GENERATOR'

   z = tcl.fully_connected(z, 4*4*1024, activation_fn=tf.identity, scope='g_z')
   z = tf.reshape(z, [batch_size, 4, 4, 1024])
   z = tcl.batch_norm(relu(z))
   
   conv1 = tcl.convolution2d_transpose(z, 512, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv1')
   #conv1 = tcl.batch_norm(relu(conv1))
   
   conv2 = tcl.convolution2d_transpose(conv1, 256, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv2')
   conv2 = tcl.batch_norm(relu(conv2))
   
   conv3 = tcl.convolution2d_transpose(conv2, 128, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv3')
   conv3 = tcl.batch_norm(relu(conv3))
   
   conv4 = tcl.convolution2d_transpose(conv3, 3, 5, 2, activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv4')
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
def netD(input_images, batch_size, reuse=False):
   print 'DISCRIMINATOR reuse = '+str(reuse)
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      conv1 = tcl.conv2d(input_images, 64, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
      conv1 = lrelu(conv1)
      
      conv2 = tcl.conv2d(conv1, 128, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
      conv2 = lrelu(conv2)

      conv3 = tcl.conv2d(conv2, 256, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
      conv3 = lrelu(conv3)

      conv4 = tcl.conv2d(conv3, 512, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4')
      conv4 = lrelu(conv4)

      #conv4_flat = tcl.flatten(conv4)
      #fc = tcl.fully_connected(conv4_flat, 1, activation_fn=tf.identity)

      conv5 = tcl.conv2d(conv4, 1, 4, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')
      return conv5

      print 'input images:',input_images
      print 'conv1:',conv1
      print 'conv2:',conv2
      print 'conv3:',conv3
      print 'conv4:',conv4
      print 'fc:',fc
      print 'END D\n'
      tf.add_to_collection('vars', conv1)
      tf.add_to_collection('vars', conv2)
      tf.add_to_collection('vars', conv3)
      tf.add_to_collection('vars', conv4)
      tf.add_to_collection('vars', fc)
      return fc

