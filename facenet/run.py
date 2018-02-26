#coding:utf-8

#执行此文件之前检查feature目录,如果更新data中照片库,则需要删除feature文件夹下feature**.txt文件。
#执行此文件，获取摄像头中的图像，若摄像头只有1个人，且连续20帧为此人，则在图像上显示姓名。
#若摄像头中两人以上，则显示每个人最相似的图片，没有连续帧率判断

"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import run_lib as rl
import cv2
import time


def run(args):
    init = rl.facenetInit()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print ('camera is not opened')
        sys.exit()
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoWriter = cv2.VideoWriter(init.videoWriterPath, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10, size)

    with tf.Graph().as_default():
        with tf.Session() as sess:

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            emb = init.getFeature(args, sess, images_placeholder, phase_train_placeholder, embeddings)
            # main
            lastPersonName = ''
            simCount = 0
            while(True):
                startTime = time.time()
                ret, frame = cap.read()
                frame_copy = frame.copy()
                total_bb,total_lm, total_face = rl.singleimg_load_and_align_data(frame, args.image_size, args.margin, args.gpu_memory_fraction)
                #print('tbb',tbb.shape,'t_',t_.shape,'tiamges',timages2.shape)
                for n in range(total_bb.shape[0]):
                    bb = total_bb[n]
                    landmark = total_lm[:,n]
                    face = total_face[n][np.newaxis,:,:,:]
                    #print('images2shape:',images2.shape)

                    feed_dict = {images_placeholder: face, phase_train_placeholder:False }
                    embcap = sess.run(embeddings, feed_dict=feed_dict)
                    dist = np.sqrt(np.sum(np.square(np.subtract(embcap[0,:], emb)),axis=1))
                    #print('  %1.4f  ' % dist, end='')
                    similarPersonName = args.image_files[np.where(dist == np.min(dist))[0][0]].split('/')[-1].split('.')[0]

                    #0.35 0.3
                    similarProb = rl.reseve_tanh(np.min(dist)) * 100
                    print('相似度:',similarProb)
                    print('摄像机中共有{0}张人脸'.format(total_bb.shape[0]))

                    frame = init.drawLandmark(frame, landmark)
                    frame = init.drawBoundingbox(frame, bb)
                    frame, simCount,lastPersonName = init.judge_N_frame(20, args, frame, total_bb, bb, dist,
                                                                         similarPersonName, similarProb,simCount, lastPersonName)
                    print ('最相似的人:',similarPersonName,'距离',np.min(dist),'概率',similarProb)

                endTime = time.time()
                fps = int(1.0/(endTime - startTime))
                #print('fps:',fps)
                frame = cv2.putText(frame, 'fps:{0}'.format(fps), (0,20), 0, 1, (0,0,255),1,False)
                cv2.imshow("Frame", frame)
                #cv2.waitKey(100)
                if init.writeFlag == 1:
                   videoWriter.write(frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    cv2.imwrite('save.jpg',frame)
                if key == ord('v'):
                    init.writeFlag = 1
                if key == ord('p'):
                    init.writeFlag = 0
                    videoWriter.release()
                if key == ord('q'):
                    break
                if key == ord('r'):
                    if total_bb.shape[0] == 1:
                        init.registCount += 1
                        frame_cropface = frame_copy[bb[1]:bb[3],bb[0]:bb[2]]
                        cv2.imwrite('./../data/faceLib_facenet/Register{0}.jpg'.format(self.registCount),frame_cropface)
                    else:
                        print('Must have only one face in Camera')

            cap.release()
            cv2.destroyAllWindows()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',default='/home/sy/facenet/20170512-110547/20170512-110547.pb')
    #parser.add_argument('--path', type=str, help='Images to compare')
    parser.add_argument('--image_files', type=str, nargs='+', help='Images to compare', default=[os.path.join('/home/sy/facenet/data/face_Guang/',imgname) for imgname in os.listdir('/home/sy/facenet/data/face_Guang/')])
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
    return parser.parse_args(argv)

if __name__ == '__main__':
    run(parse_arguments(sys.argv[1:]))
    print (sys.argv)
