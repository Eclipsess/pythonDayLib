# coding:utf-8
import cv2
import Image,ImageFont,ImageDraw
import numpy as np
frame = cv2.imread('save.jpg')
#cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil_im = Image.fromarray(frame)
draw = ImageDraw.Draw(pil_im)
#encoing选项可缺省
font = ImageFont.truetype("simhei.ttf", 20, encoding = 'unicode')
#未经过转换的通道是BGR,所以此时(0,0,255)代表R红色。
#如果经过cvtColor转换为RGB，其实没什么用，最后也要转换回来。如果在中间步骤imshow还有用。
#其中字符串要使用unicode的形式。
draw.text((20, 20), "我wowowoow我我我我我我w".decode('utf-8'), (0, 0, 255),font = font)
#frame_PIL = cv2.cvtColor(np.array(pil_im),cv2.COLOR_RGB2BGR)
cv2.imshow("Video", np.array(pil_im))
cv2.waitKey(0)
