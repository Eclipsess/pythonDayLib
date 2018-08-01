#coding=utf-8
import cv2
import numpy as np
import os
path = 'D:/coin_proj_1/validation_coin/'
total_path = os.listdir(path)

def remove_coin_bg(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (224,224))
    new_img = np.zeros([224,224],dtype='uint8')
    flag = 0
    # hough circle detection.
    circles= cv2.HoughCircles(new_img,cv2.HOUGH_GRADIENT,1,100,param1=100,param2=30,minRadius=80,maxRadius=150)
    print type(circles)
    if not circles:
        print "== None"
    #print circles.all()
    for circle in circles[0]:
        #圆的基本信息
        print(circle[2])
        #坐标行列
        x=int(circle[0])
        y=int(circle[1])
        #半径
        r=int(circle[2])
        #在原图用指定颜色标记出圆的位置
        mask=cv2.circle(new_img,(x,y),r,255,-1)
        flag = 1
        break
    if flag == 1:
        print path + " have coin circle. "
    if flag == 0:
        print path + " no coin circle. "
    # cv2.imshow('mask',mask)
    img = cv2.bitwise_and(mask, img)
    return img, flag
    # cv2.imshow('2.jpg', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
