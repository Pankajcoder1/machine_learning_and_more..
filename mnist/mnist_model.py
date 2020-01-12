# i am using tkinter to create a gui and on that write any digit
# this model will predict that digit with great accuracy

import keras
from keras.datasets import mnist
from keras.layers import Dense,MaxPooling2D,Conv2D,Dropout,Flatten
from keras.models import Sequential
from tkinter import *
import tkinter as tk
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

#this module only work fro windows
import win32gui
from PIL import ImageGrab ,Image

num_classes = 10
epochs = 10
batch_size = 128
# loading data
(X_train,y_train), (X_test,y_test) = mnist.load_data()
print("train shape and test shape\n",X_train.shape,X_test.shape)
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
input_shape = (28,28,1)
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)
# normalization process
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train = X_train/255.0
X_test = X_test/255.0
print("train shape and test shape\n",X_train.shape,X_test.shape)
print(X_train.shape[0],"train samples")
print(X_test.shape[0],"test samples")

# deining model
# you can also put all these code in a function
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Dense(128,activation = 'relu'))
model.add(Flatten())
model.add(Dense(256,activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation = "softmax"))
model.summary()
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(),metrics = ['accuracy'])
hist = model.fit(X_train,y_train, batch_size = batch_size,epochs = epochs,verbose = 1,validation_data = (X_test,y_test))
print("model is successfully trained")
model.save("mnist1.h5")
print("model is saved")
score = model.evaluate(X_test,y_test,verbose =0)
print("accuracy is ",score[1])
print("loss is ",score[0])

model = load_model("mnist.h5")

def predict_digit(img):
	img = load_img(img, grayscale = True, target_size = (28,28))
	img = img_to_array(img)
	img = img.reshape(1, 28, 28, 1)
	img = img.astype('float32')
	img = img / 255.0
	res = model.predict([img])[0]
	print(np.argmax(res), max(res))
	return np.argmax(res),max(res)

# some basic line of code to create ghui using tkinter
#and adding two button to make intresting

class App(tk.Tk):
	def __init__(self):
		tk.Tk.__init__(self)
		self.x = self.y =0
		self.canvas = tk.Canvas(self, width = 600, height = 600, bg = "white",cursor = "cross")
		self.label = tk.Label(self,text = "Thinking....",font = ("Halvetica",24) )
		self.classify_btn = tk.Button(self, text = "Recognise", command = self.classify_handwritting)
		self.button_clear = tk.Button(self,text = "clear",command = self.clear_all)

		self.canvas.grid(row = 0,column = 0,pady = 20, padx = 20, sticky = W, )
		self.label.grid(row = 0, column = 1, pady = 20, padx = 20)
		self.classify_btn.grid(row = 1, column = 1, pady = 2, padx = 2)
		self.button_clear.grid(row = 1, column = 0, pady = 2)

		self.canvas.bind("<B1-Motion>",self.draw_lines)

	def clear_all(self):
		self.canvas.delete("all")

	def classify_handwritting(self):
		#...
		HWND = self.canvas.winfo_id()
		rect = win32gui.GetWindowRect(HWND)
		im = ImageGrab.grab(rect)
		#.....
		digit,acc = predict_digit()
		self.label.configure(text = str(digit) + ',' + str(int(acc*100)) + '%')
	def draw_lines(self, event):
		self.x = event.x
		self.y = event.y
		r = 5
		self.canvas.create_oval(self.x-r,self.y-r,self.x+ r, self.y + r, fill = "black")

app = App()
mainloop()