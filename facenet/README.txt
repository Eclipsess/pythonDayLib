facenet修改自https://github.com/davidsandberg/facenet
其中src的compare.py只是在文件夹中两两比对，且每张图片无论有几张人脸，只提取出一张人脸。
修改后的文件，利用usb摄像头调取图像，实现了一张图片与图片库中的人脸对比相似度，不重复比对，并返回最相似的图片路径。
