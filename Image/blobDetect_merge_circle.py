import cv2
import numpy as np

def reduce_merge(lists):
    lists.sort()
    count = 0
    sums = 0
    record = []
    i = 0
    while(i < len(lists) -1):
        k = i
        while(lists[k] + 3 > lists[k + 1]):
            count += 1
            sums += lists[i]
            k += 1
        if count >= 10:
            av = sums / count
            record.append([av,count]) 
            sums = 0
            count = 0
        i = k      
        i += 1
    return record

def blobDetect(im):
    # Read image
    #im = cv2.imread("/home/sy/Pictures/IMG_20180207_112547.jpg", cv2.IMREAD_GRAYSCALE)
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create()
    # Detect blobs.
    keypoints = detector.detect(im)
    xy = []
    # x,w = pt[0]
    # y,h = pt[1]
    for keypoint in keypoints:
        xy.append(list(keypoint.pt))
    x = sorted([i[0] for i in xy])
    resx = reduce_merge(x)
    print resx
    y = sorted([i[1] for i in xy])
    resy = reduce_merge(y)
    print resy
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show keypoints
    cv2.namedWindow('Keypoints',0)
    cv2.resizeWindow('Keypoints', 640, 480)
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)

im = cv2.imread("/home/sy/Pictures/IMG_20180207_112547.jpg", cv2.IMREAD_GRAYSCALE)
blobDetect(im)
