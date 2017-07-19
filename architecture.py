import tensorflow as tf

from tf_ops import *


def netG(z, BATCH_SIZE):

   z = tf.layers.dense(z, 4*4*1024, name='g_z')
   z = tf.reshape(z, [BATCH_SIZE, 4, 4, 1024])
   z = tf.nn.relu(bn(z))

   conv1 = tf.layers.conv2d_transpose(z, 512, 4, strides=2, padding='SAME', name='g_conv1')
   conv1 = tf.nn.relu(bn(conv1))
   
   conv2 = tf.layers.conv2d_transpose(conv1, 256, 4, strides=2, padding='SAME', name='g_conv2')
   conv2 = tf.nn.relu(bn(conv2))

   conv3 = tf.layers.conv2d_transpose(conv2, 128, 4, strides=2, padding='SAME', name='g_conv3')
   conv3 = tf.nn.relu(bn(conv3))
   
   conv4 = tf.layers.conv2d_transpose(conv3, 3, 4, strides=2, padding='SAME', name='g_conv4')
   conv4 = tf.nn.tanh(conv4)

   return conv4

def netD(images, BATCH_SIZE, reuse=False):

   conv1 = tf.layers.conv2d(images, 64, 4, strides=2, padding='SAME', name='d_conv1', reuse=reuse)
   conv1 = lrelu(conv1)

   conv2 = tf.layers.conv2d(conv1, 128, 4, strides=2, padding='SAME', name='d_conv2', reuse=reuse)
   conv2 = lrelu(conv2)

   conv3 = tf.layers.conv2d(conv2, 256, 4, strides=2, padding='SAME', name='d_conv3', reuse=reuse)
   conv3 = lrelu(conv3)
   
   conv4 = tf.layers.conv2d(conv3, 512, 4, strides=2, padding='SAME', name='d_conv4', reuse=reuse)
   conv4 = lrelu(conv4)

   flat = tf.reshape(conv4, [BATCH_SIZE, -1])

   dense = lrelu(tf.layers.dense(flat, 1, name='d_dense', reuse=reuse))

   return dense
