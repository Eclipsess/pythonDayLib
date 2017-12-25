import cv2
from PIL import Image
personPath = './xyjy.jpg'
hatPath = './hat.png'

personImg = cv2.imread(personPath)
face_haar = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")
faces = face_haar.detectMultiScale(personImg, 1.3, 3)

personImg = Image.open(personPath)
personImg = personImg.convert('RGBA')

hatImg = Image.open(hatPath)
hatImg = hatImg.convert('RGBA')

# adjust your picture to fit Christmas hat
adapt_h = 100  
for face_x,face_y,face_w,face_h in faces:
	face_x -= face_w/2
	face_y += face_h/2
	face_w *= 2
	face_h *= 2
	hatImg = hatImg.resize((face_w, face_h))
	bg = (face_x , face_y - face_h + adapt_h, face_x + face_w, face_y + adapt_h )
	personImg.paste(hatImg, bg, mask = hatImg)

personImg.save('addHat.png')
