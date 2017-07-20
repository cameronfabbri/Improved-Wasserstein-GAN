import tensorflow as tf
from tf_ops import *

def Rk(x, channels, name, reuse=False):

   # layer 1
   conv1 = tf.layers.conv2d(x, channels, 3, strides=1, name='g_'+name, padding='SAME', reuse=reuse)
   conv1 = bn(relu(conv1))

   # layer 2
   conv2 = tf.layers.conv2d(conv1, channels, 3, strides=1, name='g'+name+'_2', padding='SAME', reuse=reuse)
   conv2 = bn(conv2)

   output = tf.add(x,conv2)
   
   return output


def netG(z,BATCH_SIZE):

   z = tf.layers.dense(z, 4*4*1024, name='g_z')
   z = tf.reshape(z, [BATCH_SIZE, 4, 4, 1024])
   z = bn(relu(z))

   conv1 = tf.layers.conv2d(z, 64, 3, strides=1, name='g_conv1', padding='SAME')
   conv1 = bn(relu(conv1))

   print 'z:',z
   print 'conv1:',conv1

   conv1 = tf.concat([conv1, conv1, conv1, conv1], axis=1)
   conv1 = tf.transpose(conv1, [0,2,3,1])
   conv1 = tf.depth_to_space(conv1, 2)
   conv1 = tf.transpose(conv1, [0,3,1,2])
   conv1 = tf.layers.conv2d(conv1, 64, 3, strides=2, name='g_conv1', padding='SAME')

   print conv1
   exit()

   r1 = Rk(conv1, 64, 'r1')
   print 'r1:',r1
   exit()




