import numpy as np
import matplotlib.pyplot as plt

# Task 1: Draw lines y = 2x + 1, y = 2x + 2, y = 2x + 3
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([-0.57, -2.57, -4.80, -7.36, -8.78, -10.52, -12.85, -14.69, -16.78])
y1 = 2*x + 1
y2 = 2*x + 2
y3 = 2*x + 3

plt.plot(x, y1, 'r--', label='y = 2x + 1', marker='+', markersize=9)
plt.plot(x, y2, 'g-.', label='y = 2x + 2', marker='+',markersize=9)
plt.plot(x, y3, 'b:', label='y = 2x + 3', marker='+',markersize=9)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Linear Equations')
plt.legend()
plt.grid()
plt.show()