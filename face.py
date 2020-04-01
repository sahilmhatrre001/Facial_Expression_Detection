import cv2
import keyboard
import numpy as np
#from skimage import io
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array


def expression(res):
	objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
	m=0.000000000000000000001
	a=res
	for i in range(0,len(a)):
		if a[i]>m:
			m=a[i]
			ind=i
	print("Expression :",objects[ind])
	return objects[ind]


model = load_model('models/expression.h5')
model.summary()
#loading haar cascade face detection model
classifier = cv2.CascadeClassifier('models/haar_cascade.xml')
# -- capture webcam
webcam = cv2.VideoCapture(0)
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = .7
fontColor              = (0,255,0)
lineType               = 2


while True:
	rate,im = webcam.read()
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	boxes = classifier.detectMultiScale(gray,1.1,8)
	if(len(boxes) != 0):
		x,y,width,height = boxes[0]
		x2,y2 = x+width,y+height
		if(keyboard.is_pressed('c')):
			face = im[y:y+height,x:x+width]
			cv2.imwrite("temp.png".format(0),face)
			face_img = load_img("temp.png",color_mode = "grayscale",target_size=(48,48))
			xi = img_to_array(face_img)
			xi = np.expand_dims(xi,axis = 0)
			print("==============================")
			print(xi)
			print("==============================")
			xi /= 255
			print(xi)
			print("==============================")
			custom = model.predict(xi);
			print(custom[0])
			result = expression(custom[0])
			cv2.putText(face,result, 
			(10,50), 
			font, 
			fontScale,
			fontColor,
			lineType)	
			cv2.imshow('captured face',face)
		elif(keyboard.is_pressed('x')):
			webcam.release()
			cv2.destroyAllWindows()
		cv2.rectangle(im,(x,y),(x2,y2),(0,0,255),5)
		cv2.putText(im,'Press c to capture face', 
			(10,450), 
			font, 
			fontScale,
			fontColor,
			lineType)
		cv2.putText(im,'Press x to exit', 
			(10,470), 
			font, 
			fontScale,
			fontColor,
			lineType)
		cv2.imshow('detected faces',im)
		cv2.waitKey(1)