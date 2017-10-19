import scipy.misc as misc
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import numpy as np
import random
import ntpath
import sys
import cv2
import os
from skimage import color
import argparse
import data_ops
from tf_ops import *
import gzip
import mnist_reader


'''
   Batch norm before relu
'''
def netG(z, batch_size):
   print 'GENERATOR'

   z = tcl.fully_connected(z, 4*4*256, activation_fn=tf.identity, scope='g_z')
   z = tf.reshape(z, [batch_size, 4, 4, 256])
   z = tcl.batch_norm(z)
   z = tf.nn.relu(z)
   
   conv1 = tcl.convolution2d_transpose(z, 128, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv1')

   conv2 = tcl.convolution2d_transpose(conv1, 64, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv2')

   conv3 = tcl.convolution2d_transpose(conv2, 1, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv3')

   conv3 = conv3[:,:28,:28,:]

   print 'z:',z
   print 'conv1:',conv1
   print 'conv2:',conv2
   print 'conv3:',conv3
   return conv3 

'''
   Discriminator network.
'''
def netD(input_images, batch_size, SELU, NORM, reuse=False):

   print 'DISCRIMINATOR reuse = '+str(reuse)
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      conv1 = tcl.conv2d(input_images, 64, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
      if NORM: conv1 = tcl.layer_norm(conv1)
      if SELU: conv1 = selu(conv1)
      else:    conv1 = lrelu(conv1)
      
      conv2 = tcl.conv2d(conv1, 128, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
      if NORM: conv2 = tcl.layer_norm(conv2)
      if SELU: conv2 = selu(conv2)
      else:    conv2 = lrelu(conv2)

      conv3 = tcl.conv2d(conv2, 256, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
      if NORM: conv3 = tcl.layer_norm(conv3)
      if SELU: conv3 = selu(conv3)
      else:    conv3 = lrelu(conv3)

      conv4 = tcl.conv2d(conv3, 1, 4, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')

      print 'input images:',input_images
      print 'conv1:',conv1
      print 'conv2:',conv2
      print 'conv3:',conv3
      print 'conv4:',conv4
      print 'END D\n'

      return conv4

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--DATASET',    required=True,help='The DATASET to use')
   parser.add_argument('--DATA_DIR',   required=True,help='Directory where data is')
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size',type=int,default=128)
   parser.add_argument('--NORM',       required=False,help='Use layer normalization in D',type=int,default=0)
   parser.add_argument('--SELU',       required=False,help='Use SELU',type=int,default=0)
   parser.add_argument('--SCALE',      required=False,help='Scale of gradient penalty',type=int,default=10)
   parser.add_argument('--MAX_STEPS',  required=False,help='How long to train',type=int,default=1000000)
   a = parser.parse_args()

   DATASET        = a.DATASET
   DATA_DIR       = a.DATA_DIR
   BATCH_SIZE     = a.BATCH_SIZE
   SCALE          = a.SCALE
   NORM           = bool(a.NORM)
   SELU           = bool(a.SELU)
   MAX_STEPS      = a.MAX_STEPS

   CHECKPOINT_DIR = 'checkpoints/DATASET_'+DATASET+'/SCALE_'+str(SCALE)+'/NORM_'+str(NORM)+'/SELU_'+str(SELU)+'/'
   IMAGES_DIR     = CHECKPOINT_DIR+'images/'
   
   try: os.makedirs(IMAGES_DIR)
   except: pass


   # open mnist fashion

   X_train, y_train = mnist_reader.load_mnist(DATA_DIR, kind='train')
   X_test, y_test   = mnist_reader.load_mnist(DATA_DIR, kind='t10k')
  
   train_images = np.empty((70000, 28, 28, 1), dtype=np.float32)

   i = 0
   for img in X_train:
      img = np.reshape(img, (28, 28, 1)).astype('float32')
      train_images[i, ...] = img
      i += 1
   
   for img in X_test:
      img = np.reshape(img, (28, 28, 1)).astype('float32')
      train_images[i, ...] = img
      i += 1

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')

   real_images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28, 28, 1), name='real_images')

   # generated images
   gen_images = netG(z, BATCH_SIZE)

   # get the output from D on the real and fake data
   errD_real = netD(real_images, BATCH_SIZE, SELU, NORM)
   errD_fake = netD(gen_images, BATCH_SIZE, SELU, NORM, reuse=True)

   # cost functions
   errD = tf.reduce_mean(errD_real) - tf.reduce_mean(errD_fake)
   errG = tf.reduce_mean(errD_fake)

   # gradient penalty
   epsilon = tf.random_uniform([], 0.0, 1.0)
   x_hat = real_images*epsilon + (1-epsilon)*gen_images
   d_hat = netD(x_hat, BATCH_SIZE, SELU, NORM, reuse=True)
   gradients = tf.gradients(d_hat, x_hat)[0]
   slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
   gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
   errD += gradient_penalty

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   # optimize G
   G_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.0,beta2=0.9).minimize(errG, var_list=g_vars, global_step=global_step)

   # optimize D
   D_train_op = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.0,beta2=0.9).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)

   summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR+'/'+'logs/', graph=tf.get_default_graph())

   tf.add_to_collection('G_train_op', G_train_op)
   tf.add_to_collection('D_train_op', D_train_op)
   
   # restore previous model if there is one
   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   ########################################### training portion

   step = sess.run(global_step)
   
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)

   n_critic = 5

   while step < MAX_STEPS:
      
      start = time.time()

      # train the discriminator for 5 or 25 runs
      for critic_itr in range(n_critic):
         batch_images = random.sample(train_images, BATCH_SIZE)
         batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         sess.run(D_train_op, feed_dict={z:batch_z, real_images:batch_images})

      # now train the generator once! use normal distribution, not uniform!!
      batch_images = random.sample(train_images, BATCH_SIZE)
      batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
      sess.run(G_train_op, feed_dict={z:batch_z, real_images:batch_images})

      # now get all losses and summary *without* performing a training step - for tensorboard
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={z:batch_z, real_images:batch_images})
      summary_writer.add_summary(summary, step)

      print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',time.time()-start
      step += 1
    
      if step%1000 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
         batch_z  = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_images = random.sample(train_images, BATCH_SIZE)
         gen_imgs = np.asarray(sess.run([gen_images], feed_dict={z:batch_z, real_images:batch_images}))[0]
         random.shuffle(gen_imgs)

         for c in range(0,5):
            plt.imsave(CHECKPOINT_DIR+'images/0000'+str(step)+'_'+str(c)+'.png', np.squeeze(gen_imgs[c]), cmap=plt.cm.gray)
         print 'Done saving'



