def rotateDegree(img, degree):
    (h,w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, degree, 1.0)
    rotateImg = cv2.warpAffine(img, M, (w,h))
    return rotateImg
    
