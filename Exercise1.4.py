import numpy as np

A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])

A_inv = np.linalg.inv(A)

x = np.dot(A, A_inv)
y = np.dot(A_inv, A)

print("A * A^-1 :")
print(x)
print("A^-1 * A :")
print(y)