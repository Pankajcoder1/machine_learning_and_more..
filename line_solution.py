#this is show the conditon of two line.
import numpy as np
import matplotlib.pyplot as plt
print("enter the slope and intercept of first line.")
m1=int(input())
c1=int(input())
print("enter slope and intercept of second line.")
m2=int(input())
c2=int(input())
print("enter the range of x variable.(like -x to x).")
a=int(input())
b=int(input())
x=np.arange(a,b)
plt.xlabel("value of x.")
plt.ylabel("line")
if(m1==m2):
	print("you enter equation of two parallel line.")
else:
	print("Intesection will be solution.")
plt.title("Intesection represent solution")
plt.plot(x,m1*x+c1,label="line1",color="red")
plt.plot(x,m2*x+c2,label="line2",color="green")
plt.legend()
plt.show()