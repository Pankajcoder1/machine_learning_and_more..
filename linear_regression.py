import numpy as np
import matplotlib.pyplot as plt
def coefficient(x,y):
	n=np.size(x)
	m_x=np.mean(x)
	m_y=np.mean(y)
	ss_xy=np.sum(y*x)-n*m_y*m_x
	ss_xx=np.sum(x*x)-n*m_x*m_x
	b1=ss_xy/ss_xx
	b0=m_y-b1*m_x
	return (b0,b1)
def main():
	# observations 
	x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
	y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
	b=coefficient(x,y)
	plt.scatter(x,y,marker="o",color="red",label="scatter_point")
	y=b[0]+b[1]*x
	plt.plot(x,y,label="line_of_regression",color="green")
	plt.ylim(0,20)
	plt.xlabel("value_of_X")
	plt.ylabel("dependent_value")
	plt.title("made by pankaj kumar.")
	plt.legend()
	plt.show()

main()