import cv2 as cv2
import numpy as numpy

print("Enter name of file with extension")
str1=input()
print("press 's' to save file")
face_detect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#enter path of cascade file of face detection
obj=cv2.VideoCapture(0) 
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
out=cv2.VideoWriter(str1,fourcc,20.0,(640,480))
while(True):
	check, frame=obj.read()
	face=face_detect.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=5,minSize=(50,50))
	#here scalefctor is 1.3 to detect actuall face .
	#value depend on photo 
	for (x,y,w,h) in face:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,100),2)
	if(check == True):
		cv2.imshow(str1,frame)
		out.write(frame) #to save in file.
		k=cv2.waitKey(1) & 0xFF
		if k==ord('s'):
			break ;
#give mesaage
if k==ord('s'):
	print("video is saved")
else :
	print("video is not saved")
obj.release()
out.release()
cv2.destroyAllWindows()