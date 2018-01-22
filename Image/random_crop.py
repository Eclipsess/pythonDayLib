import cv2
import tensorflow as tf
import numpy as np

im1 = cv2.imread('/home/sy/keyboard_step2/data/threshold/0/2018201813174189461.jpg',0)[:,:,np.newaxis]
print im1.shape
img = tf.placeholder(tf.int32, shape = (None,None,1))
image = tf.random_crop(img, [int(0.8*im1.shape[0]),int(0.8*im1.shape[1]),1])

with tf.Session() as sess:
    for i in range(10):
        feed = {img:im1}
        show = sess.run(image,feed_dict=feed)
        np.asarray(show,dtype=np.int8)
        print show.shape
        cv2.imwrite('{0}.jpg'.format(i),show)
