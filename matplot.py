import matplotlib.pyplot as plt  

x = [2,4,6,8,10]
y = [6,3,7,4,6]

x1 = [1,3,5,7,9]
y1 = [3,9,2,7,4]


plt.plot(x,y,color="red",label="One")
plt.plot(x1,y1,color="green",label="Two")
plt.xlabel("X-Axis")
plt.ylabel("Y-Axis")
plt.title("Graph\n")
plt.legend()
plt.show()
