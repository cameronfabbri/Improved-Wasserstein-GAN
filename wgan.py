import tensorflow as tf
import tensorflow.contrib.layers as layers
import sys

'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2, name='lrelu'):
   return tf.maximum(leak*x, x)

def netG(z, batch_size):
   print 'GENERATOR'
   z = layers.fully_connected(z, 4*4*1024, normalizer_fn=layers.batch_norm, activation_fn=tf.identity, scope='g_z')
   z = tf.reshape(z, [batch_size, 4, 4, 1024])
   
   conv1 = layers.convolution2d_transpose(z, 512, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=tf.nn.relu, scope='g_conv1')
   conv2 = layers.convolution2d_transpose(conv1, 256, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=tf.nn.relu, scope='g_conv2')
   conv3 = layers.convolution2d_transpose(conv2, 128, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=tf.nn.relu, scope='g_conv3')
   conv4 = layers.convolution2d_transpose(conv3, 3, 5, stride=2, activation_fn=tf.nn.tanh, scope='g_conv4')
   
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
      conv1 = layers.conv2d(input_images, 64, 5, stride=2, activation_fn=None, scope='d_conv1')
      conv1 = lrelu(conv1)
      conv2 = layers.conv2d(conv1, 128, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=None, scope='d_conv2')
      conv2 = lrelu(conv2)
      conv3 = layers.conv2d(conv2, 256, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=None, scope='d_conv3')
      conv3 = lrelu(conv3)
      conv4 = layers.conv2d(conv3, 512, 5, stride=2, normalizer_fn=layers.batch_norm, activation_fn=None, scope='d_conv4')
      conv4 = lrelu(conv4)
      conv5 = layers.conv2d(conv4, 1, 4, stride=1, activation_fn=None, scope='d_conv5')
      conv5 = lrelu(conv5)

      print 'input images:',input_images
      print 'conv1:',conv1
      print 'conv2:',conv2
      print 'conv3:',conv3
      print 'conv4:',conv4
      print 'conv5:',conv5
      print 'END D\n'
      tf.add_to_collection('vars', conv1)
      tf.add_to_collection('vars', conv2)
      tf.add_to_collection('vars', conv3)
      tf.add_to_collection('vars', conv4)
      tf.add_to_collection('vars', conv5)
      return conv5

