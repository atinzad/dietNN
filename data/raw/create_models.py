#Create Model and save in Tensorflow via Keras

from keras import backend as K
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import os




sess = tf.Session()
K.set_session(sess)

model = VGG16()
saver = tf.train.Saver()
save_path = saver.save(sess, os.getcwd() + "/model.ckpt")
