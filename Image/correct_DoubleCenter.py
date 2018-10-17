import cv2
import math
# year correct in coin, link coin center and ssd->year center and correct angle 
def correctYear(xmin, xmax, ymin, ymax, centerXX, centerYY):
    deltaX = centerXX - (xmin + xmax) / 2
    deltaY = centerYY - (ymin + ymax) / 2
    degree = math.atan(float(deltaX)/(deltaY+0.000001))*(float(180)/math.pi)
    return -degree

img = cv2.imread("moto.jpg")

xmin,xmax,ymin,ymax = 300,400,200,250
img2 = img[ymin:ymax, xmin:xmax]     # year bbox

cv2.imshow('moto2.jpg',img2)
cv2.waitKey(0)

centerX = (xmin + xmax) / 2
centerY = (ymin + ymax) / 2
centerYY = img.shape[0] / 2
centerXX = img.shape[1] / 2
X_thresh = 10

img2_cX = img2.shape[1]/2
img2_cY = img2.shape[0]/2
center=(img2_cX,img2_cY)
if abs(centerX - centerXX) > X_thresh:
    degree = correctYear(xmin, xmax, ymin, ymax, centerXX, centerYY)
    print "degree: ", degree
    if degree != 0:
        rot_mat = cv2.getRotationMatrix2D(center=(img2_cX,img2_cY), angle=degree, scale=1)
        rot_img = cv2.warpAffine(img2, rot_mat, (img2.shape[1], img2.shape[0]),borderValue=(255,255,255))
        #bbox = cv2.RotatedRect(center, cv2.Size(img.cols, img.rows), degree).boundingRect()
        cv2.imshow('show',rot_img)
        # cv2.imwrite('./1.jpg',rot_img)
        cv2.waitKey(0)
