import tensorflow as tf
import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import os
import fnmatch
import cPickle as pickle
import scipy.misc as misc


def _read_input(filename_queue):
   class DataRecord(object):
      pass
   reader             = tf.WholeFileReader()
   key, value         = reader.read(filename_queue)
   record             = DataRecord()
   decoded_image      = tf.image.decode_jpeg(value, channels=3)
   decoded_image_4d   = tf.expand_dims(decoded_image, 0)
   resized_image      = tf.image.resize_bilinear(decoded_image_4d, [96, 96])
   record.input_image = tf.squeeze(resized_image, squeeze_dims=[0])
   cropped_image      = tf.cast(tf.image.central_crop(decoded_image, 0.6), tf.float32)
   decoded_image_4d   = tf.expand_dims(cropped_image, 0)
   resized_image      = tf.image.resize_bilinear(decoded_image_4d, [64, 64])
   record.input_image = tf.squeeze(resized_image, squeeze_dims=[0])
   return record


def read_input_queue(filename_queue, batch_size):
   read_input = _read_input(filename_queue)
   num_preprocess_threads = 8
   min_queue_examples = int(0.1 * 100)
   print("Shuffling")
   input_image = tf.train.shuffle_batch([read_input.input_image],
                                        batch_size=batch_size,
                                        num_threads=num_preprocess_threads,
                                        capacity=min_queue_examples + 8 * batch_size,
                                        min_after_dequeue=min_queue_examples)
   input_image = input_image/127.5 - 1.
   return input_image


def saveImage(images, step, image_dir):
   num = 0
   for image in images:
      image = (image+1.)
      image *= 127.5
      image = np.clip(image, 0, 255).astype(np.uint8)
      image = np.reshape(image, (64, 64, -1))
      misc.imsave(image_dir+str(step)+'_'+str(num)+'.jpg', image)
      num += 1
      if num == 5:
         break

'''
   Inputs: A directory containing images (can have nested dirs inside) and optional extension
   Outputs: A list of image paths
'''
def getPaths(data_dir, ext='jpg'):
   pattern   = '*.'+ext
   image_list = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            image_list.append(os.path.join(d,filename))
   return image_list

'''
   Loads the image paths
'''
def loadData(data_dir, dataset):
   
   pkl_file = dataset+'.pkl'
   if os.path.isfile(pkl_file):
      print 'Pickle file found'
      image_paths = pickle.load(open(pkl_file, 'rb'))
      return image_paths
   else:
      print 'Getting paths!'
      image_paths = getPaths(data_dir)
      pf   = open(pkl_file, 'wb')
      data = pickle.dumps(image_paths)
      pf.write(data)
      pf.close()
      return image_paths
