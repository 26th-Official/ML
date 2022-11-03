from statistics import mean
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use("ggplot")

x = np.array([1,2,3,4,5,6], dtype=np.float64)
y = np.array([5,4,6,5,6,7], dtype=np.float64)

def generate_df(count,variance,step = 2,correl=False):
    start = 1
    y = []
    for i in range(count):
        y.append(start + (random.randrange(-variance,variance)))
        if correl and correl == "pos":
            start += step
        elif correl and correl == "neg":
            start -= step
    x = [i for i in range(len(y))] 
    return np.array(x,dtype=np.float64),np.array(y,dtype=np.float64)


x,y = generate_df(40,10,2,correl="pos")

def slope_intercept(x,y):
    m = ((mean(x)*mean(y)) - (mean(x*y)))/ (((mean(x))**2) - (mean(x**2)))
    b = mean(y) - (m*mean(x))
    return m,b

def r_square(y,y_cap):
    r = 1 - ((sum((y - y_cap)**2))/(sum((y - mean(y))**2)))
    return r

m,b = slope_intercept(x,y)
regression_line = [(m*i)+b for i in x]


print(r_square(y,regression_line))

plt.scatter(x,y)
plt.plot(x,regression_line,color="b")
plt.show()
