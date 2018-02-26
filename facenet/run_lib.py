# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face
import Image,ImageFont,ImageDraw
import cv2
import time

class facenetInit(object):

    def __init__(self, sim_threshold = 60.0, videoWriterPath = 'MyOutputVid.mp4',writeFlag = 0,registCount = 0,
                 featurePath = './feature/feature_128_Guang.txt',
                 ):
        self.sim_threshold = sim_threshold
        self.videoWriterPath = videoWriterPath
        self.writeFlag = writeFlag
        self.registCount = 0
        self.featurePath = featurePath

    def getFeature(self, args, sess, images_placeholder, phase_train_placeholder, embeddings):
        if os.path.exists(self.featurePath):
            emb = np.loadtxt(self.featurePath)
        else:
            images = load_and_align_data(args.image_files, args.image_size, args.margin,
                                         args.gpu_memory_fraction)
            emb = np.array(np.zeros([images.shape[0], 128]))
            for i in range(images.shape[0]):
                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: images[i][np.newaxis, :, :, :], phase_train_placeholder: False}
                emb[i] = sess.run(embeddings, feed_dict=feed_dict)
            np.savetxt(self.featurePath, emb)
        return emb

    def drawLandmark(self, frame, landmark):
        for i in range(5):
            frame = cv2.circle(frame, (landmark[i], landmark[i + 5]), 5, (0, 0, 255), -1)
        return frame

    def drawBoundingbox(self, frame, bb):
        frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)
        return frame

    def addChineseInit(self, frame):
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        draw = ImageDraw.Draw(pil_im)
        font = ImageFont.truetype("simhei.ttf", 30, encoding="utf-8")
        return draw, font, pil_im

    def judge_N_frame(self,n,args,frame,total_bb,bb,dist,similarPersonName,similarProb,simCount,lastPersonName):

        draw, font, pil_im = self.addChineseInit(frame)

        if total_bb.shape[0] == 1:
            print('开启单人连续20帧判断模式')
            if similarProb > self.sim_threshold:
                print('上一帧大于60%', lastPersonName)
                print('连续{0}帧判断是此人'.format(simCount))
                if simCount >= n and lastPersonName == similarPersonName:
                    simCount += 1
                    print(simCount)
                    draw.text((bb[0], bb[1]), (similarPersonName + (' %.1f %%') % similarProb).decode('utf-8'),
                              (255, 255, 0), font=font)
                else:
                    print(lastPersonName, similarPersonName)
                    if lastPersonName == similarPersonName:
                        simCount += 1
                    else:
                        simCount = 0
                lastPersonName = args.image_files[np.where(dist == np.min(dist))[0][0]].split('/')[-1].split('.')[0]
                print('上一个人的名字:', lastPersonName)
            else:
                simCount = 0
                draw.text((bb[0], bb[1]), '陌生人 {0}'.format('%.1f %%' % similarProb).decode('utf-8'), (0, 0, 255),
                          font=font)
        else:
            if similarProb > self.sim_threshold:
                draw.text((bb[0], bb[1]), (similarPersonName + (' %.1f %%') % similarProb).decode('utf-8'),
                          (255, 255, 0), font=font)
            else:
                draw.text((bb[0], bb[1]), '陌生人 {0}'.format('%.1f %%' % similarProb).decode('utf-8'), (0, 0, 255),
                          font=font)
        frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        return frame, simCount, lastPersonName

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

def singleimg_load_and_align_data(img, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print (img)
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    # print('bouding',bounding_boxes.shape,'_',_.shape)
    bb = []
    img_list = []
    tbb = []
    for i in range(bounding_boxes.shape[0]):
        det = np.squeeze(bounding_boxes[i, 0:4])
        # 下一句要放在循环内，要不然会出现赋值第二个bb的时候，改变第一个bb，导致tbb存放的两个bb都相同。
        # 也可以通过浅拷贝或者深拷贝的方法。
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2]]
        aligned = cv2.resize(cropped, (image_size, image_size))
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        tbb.append(bb)
    tbb = np.array(tbb)
    if img_list:
        images = np.stack(img_list)
    else:
        images = np.array([])
    # print ('tbb.shape',bb.shape,'iamges.shape',images.shape)
    return tbb, _, images

def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in range(nrof_samples):
        img = misc.imread(os.path.expanduser(image_paths[i]), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if bounding_boxes.shape[0] != 1:
            print(image_paths[i])
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images

def reseve_tanh(x):
    sinh = (np.exp(x) - np.exp(-x)) / 2
    cosh = (np.exp(x) + np.exp(-x)) / 2
    tanh = sinh / cosh
    return 1 - tanh
