import cPickle as pickle
import tensorflow as tf
from scipy import misc
import numpy as np
import argparse
import ntpath
import sys
import os
import time
import imageio

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'nets/')

from tf_ops import *
import data_ops
from architecture import *

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--ARCHITECTURE', required=False,default='dcgan',type=str,help='Architecture to use')
   parser.add_argument('--DATASET',      required=False,default='celeba',help='The dataset to use')
   parser.add_argument('--LEARNING_RATE',required=False,default=0.0001,type=float,help='Learning rate for the pretrained network')
   parser.add_argument('--BATCH_SIZE',   required=False,default=64,type=int,help='Batch size to use')
   parser.add_argument('--SELU',         required=False,default=0,type=int,help='Whether or not to use SELU')
   a = parser.parse_args()

   ARCHITECTURE  = a.ARCHITECTURE
   DATASET       = a.DATASET
   LEARNING_RATE = a.LEARNING_RATE
   BATCH_SIZE    = a.BATCH_SIZE

   EXPERIMENT_DIR = 'checkpoints/ARCHITECTURE_'+ARCHITECTURE+'/DATASET_'+DATASET+'/'
   
   IMAGES_DIR = EXPERIMENT_DIR+'images/'

   DATA_DIR='/mnt/data2/images/celeba/images/'

   print
   print 'Creating',EXPERIMENT_DIR
   try: os.makedirs(IMAGES_DIR)
   except: pass
   
   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # placeholder for z
   z = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')

   train_images_list = data_ops.loadData(DATA_DIR, DATASET)

   real_images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 64, 64, 3), name='real_images')
   filename_queue    = tf.train.string_input_producer(train_images_list)
   real_images       = data_ops.read_input_queue(filename_queue, BATCH_SIZE)

   # generate images given z
   gen_images = netG(z, BATCH_SIZE)
   
   # D's decision on real data
   D_real = netD(real_images, BATCH_SIZE)

   # D's decision on generated data
   D_fake = netD(gen_images, BATCH_SIZE, reuse=True)

   errD = tf.reduce_mean(D_fake)-tf.reduce_mean(D_real)
   errG = -tf.reduce_mean(D_fake)
   
   # gradient penalty stuff
   epsilon = tf.random_uniform([BATCH_SIZE, 1,1,1], 0.0, 1.0)
   x_hat = real_images*epsilon + (1-epsilon)*gen_images
   d_hat = netD(x_hat, BATCH_SIZE, reuse=True)
   gradients = tf.gradients(d_hat, x_hat)[0]
   slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
   gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
   errD += gradient_penalty

   # tensorboard summaries
   try: tf.summary.scalar('d_loss', tf.reduce_mean(errD))
   except:pass
   try: tf.summary.scalar('g_loss', tf.reduce_mean(errG))
   except:pass

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   G_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,beta1=0.5,beta2=0.9).minimize(errG, var_list=g_vars, global_step=global_step)
   D_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,beta1=0.5,beta2=0.9).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)
   
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess = tf.Session()
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(EXPERIMENT_DIR+'/logs/', graph=tf.get_default_graph())

   tf.add_to_collection('vars', G_train_op)
   tf.add_to_collection('vars', D_train_op)

   ckpt = tf.train.get_checkpoint_state(EXPERIMENT_DIR)
   # restore previous model if there is one
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
   merged_summary_op = tf.summary.merge_all()

   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)

   while True:

         batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         sess.run([G_train_op], feed_dict={z:batch_z})

         for itr in xrange(5):
            batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
            sess.run([D_train_op], feed_dict={z:batch_z})

         batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={z:batch_z})

         summary_writer.add_summary(summary, step)
         print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss
         step += 1
         
         if step%100 == 0:
            print 'Saving model...'
            saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
            saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
            print 'Model saved\n'

            batch_z  = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
            gen_imgs = sess.run([gen_images], feed_dict={z:batch_z})

            data_ops.saveImage(gen_imgs[0], step, IMAGES_DIR)
            print 'Done saving'

