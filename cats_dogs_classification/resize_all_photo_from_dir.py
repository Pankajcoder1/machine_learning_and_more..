#this code resize all the image of file.
import cv2 as cv2
import matplotlib.pyplot as plt
from os import listdir
folder = "data_set/test/cats/"
#enter path of file
for file in listdir(folder):
	photo = cv2.imread(folder + file)
	photo = cv2.resize(photo,(200,200))
	cv2.imwrite(folder+file,photo)
	# cv2.imshow("pk",photo)
	# cv2.waitKey(100)
	#this show method make program slow.
print("done")