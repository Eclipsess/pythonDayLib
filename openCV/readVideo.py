import cv2
path = '/home/sy/face_Identify/detection/code/demo_0.4_0.8.mp4'
video = cv2.VideoCapture(path)
totalframe = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print totalframe
fps = int(video.get(cv2.CAP_PROP_FPS))
print fps
startfps = 100
endfps = 200
video.set(cv2.CAP_PROP_POS_FRAMES, startfps)

videoWriterPath = 'demo_1229_dav.mp4'
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

print size
videoWriter = cv2.VideoWriter(videoWriterPath, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 25, size)

writeFlag = 0

while (startfps < endfps):
    startfps += 1
    ret, frame = video.read()
    print ret
    cv2.imshow('frame',frame)
    cv2.waitKey(40)
    #break
    if writeFlag == 1:
        videoWriter.write(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite('save.jpg',frame)
    if key == ord('v'):
        writeFlag = 1
    if key == ord('p'):
        writeFlag = 0
        videoWriter.release()
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
