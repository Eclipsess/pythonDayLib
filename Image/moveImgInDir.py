#coding:utf-8
import os
import shutil
path = '0.人脸识别照片'
keyword = ['jpg','png','JPG','PNG','JPEG','jpeg']
objpath = '/home/sy/facenet/data/face_Guang'
count = 0
for dirpath, dirname, filename in  os.walk(path):
    for name in filename:
        for key in keyword:
            if name.endswith(key):
                shutil.copy(os.path.join(dirpath, name), os.path.join(objpath, name)) 
                count += 1
                break
print 'total Pic:', count
print 'finished'
