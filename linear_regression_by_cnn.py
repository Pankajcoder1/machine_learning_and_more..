import keras
from keras import Sequential
from keras.layers import Dense,Conv2D
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD
import matplotlib.pyplot as plt

#you can put any add here for linear regression.
Fahrenheit=np.array([-140,-136,-124,-112,-105,-96,-88,-75,-63,-60,-58,-40,-20,-10,0,30,35,48,55,69,81,89,95,99,105,110,120,135,145,158,160],dtype=float)
Celsius=np.array([-95.55,-93.33,-86.66,-80,-76.11,-71.11,-66.66,-59.44,-52.77,-51.11,-50,-40,-28.88,-23.33,-17.77,-1.11,1.66,8.88,12,20,27.22,31.66,35,37.22,40.55,43.33,48.88,57.22,62.77,70,71.11],dtype=float)

#model definition.
model = Sequential()
model.add(Dense(64,activation = "relu",input_shape = [1]))
model.add(Dense(128,activation = "relu"))
model.add(Dense(1)) 
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
model.fit(Fahrenheit,Celsius,epochs=10000)
value=int(input("enter the value of Fahrenheit for predict: "))
ans=model.predict([value])
print(ans)
#plotting value.
plt.title("linear regression")
plt.xlabel("Fahrenheit")
plt.ylabel("Celsius")
plt.plot(Fahrenheit,Celsius,label="training data",color = "blue")
plt.scatter(value,ans,label="predicted value",color="red")
plt.legend()
plt.show()