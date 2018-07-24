import cv2

img = cv2.imread('./xxx.jpg')
img_leftmirror = cv2.transpose(img) #左旋90+镜像
img_left = cv2.flip(cv2.transpose(img),0) #左旋90
img_right = cv2.flip(cv2.transpose(img),1) #右旋90
img_rightmirror = cv2.flip(cv2.transpose(img),-1) #右旋90+镜像
