from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt

def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(200, 200))
	img = img_to_array(img)
	img = img.reshape(1, 200, 200, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img
 

def run_example():
	file_name = input("enter name of file")
	img = load_image(file_name)
	model = load_model("pankaj.h5")
	# predict the class
	result = model.predict(img)
	c=result[0]
	c = int(c)
	if(c == 1):
		print("this is a dog.")
	if(c == 0):
		print("this is a cat")
	else:
		print("something else")

	plt.title("image")
	img = plt.imread(file_name)
	plt.imshow(img)

	plt.show()

run_example()