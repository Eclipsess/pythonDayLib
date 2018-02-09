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
        while(lists[k] + 10 > lists[k + 1] and k < (len(lists) - 2)):
            count += 1
            sums += lists[i]
            k += 1
        if count >= 12:
            av = sums / count
            record.append([av,count]) 
        count = 0
        sums = 0
        count = 0
        i = k      
        i += 1
    return record

def blobDetect(im):
    # Read image
    #im = cv2.imread("/home/sy/Pictures/IMG_20180207_112547.jpg", cv2.IMREAD_GRAYSCALE)
    # Set up the detector with default parameters.
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200


    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 1400

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)
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

im = cv2.imread("/home/sy/Pictures/IMG_20180207_111956.jpg", cv2.IMREAD_GRAYSCALE)

#im = cv2.GaussianBlur(im, (5,5), 3)
blobDetect(im)
