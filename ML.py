import matplotlib.pyplot as plt
import numpy as np
def f(x):
    return 2*x**2
x = np.array(range(5))
y = f(x)
print(x)
print(y)

plt.plot(x, y)
plt.show()