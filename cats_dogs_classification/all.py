import numpy as np
import cv2 as cv2
import sys
from os import listdir
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

#replace of path of dogs folder of train subfolder.
dog_folder = "project/data_set/train/dogs/"
print("first few images of dog and cat")
# plot first few images
for i in range(9):
	plt.subplot(330 + 1 + i)
	dog_name = dog_folder + "dog." + str(i) + ".jpg"
	dog_image = plt.imread(dog_name)
	plt.imshow(dog_image)

plt.show()
#replace of path of cats folder of train subfolder.
cat_folder = "project/data_set/train/cats/"
for i in range(9):
	plt.subplot(330 + 1 + i)
	cat_name = cat_folder + "cat." + str(i) + ".jpg"
	cat_image = plt.imread(cat_name)
	plt.imshow(cat_image)
plt.show()

#load dogs vs cats dataset, reshape and save to a new file
#define location of dataset
folder = "project/data_set/train/cats/"
photos, labels = list(), list()
# enumerate files in the directory
for file in listdir(folder):
	# determine class
	output = 0.0
	if file.startswith("cat"):
		output = 1.0
	# load image
	photo = load_img(folder + file, target_size=(200, 200))
	# convert to numpy array
	photo = img_to_array(photo)
	# store
	photos.append(photo)
	labels.append(output)
# convert to a numpy arrays
photos = np.asarray(photos)
labels = np.asarray(labels)
print(photos.shape, labels.shape)
# save the reshaped photos
np.save("dogs_vs_cats_photos.npy", photos)
np.save("dogs_vs_cats_labels.npy", labels)


# define cnn model
def define_model():
	model = Sequential()
	#first layer.
	#if your have ram more than 8gb then never run this is all layer

	model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	# model.summary()
	# #second layer
	model.add(Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	# # model.summary()
	#this used in last otherwise i face an error.
	# # #third model
	# model.add(Conv2D(128, (3, 3), activation="relu", padding="same", input_shape=(200, 200, 3)))
	# model.add(MaxPooling2D((2, 2)))
	# model.add(Dropout(0.2))
	# # #third layer
	# model.add(Conv2D(256, (3, 3), activation="relu", padding="same", input_shape=(200, 200, 3)))
	# model.add(MaxPooling2D((2, 2)))
	# model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(1, activation="sigmoid"))
	# compile model
	opt = SGD(lr = 0.001, momentum=0.9)
	model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
	return model

# plot observation curve
def summarize_diagnostics(history):
	# plot loss
	plt.subplot(211)
	plt.title("Cross Entropy Loss")
	plt.plot(history.history["loss"], color="blue", label="train")
	plt.plot(history.history["val_loss"], color="orange", label="test")
	plt.legend()
	# plot accuracy
	plt.subplot(212)
	plt.title("Classification Accuracy")
	plt.plot(history.history["accuracy"], color="blue", label="train")
	plt.plot(history.history["val_accuracy"], color="orange", label="test")
	plt.legend()
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	plt.savefig(filename + "_plot.png")
	#you can change name of file
	plt.show()
	plt.close()
 

def run_test_harness():
	model = define_model()
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = datagen.flow_from_directory("project/data_set/train",
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = datagen.flow_from_directory("project/data_set/test",
		class_mode='binary', batch_size=64, target_size=(200, 200))
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
	_, acc = model.evaluate_generator(test_it, 1000/64 ,workers=12)
	print('> %.3f' % (acc * 100.0))

	# learning curves
	summarize_diagnostics(history)
	model.summary()
	model.save("pankaj.h5")
 
run_test_harness()

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