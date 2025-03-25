import numpy as np
from numpy.ma.core import multiply

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
add = a+b
print(add)

sub = a-b
print(sub)

multi = a@b
print(multi)

div = a/b
print(div)

sin = np.sin(a)
print(sin)

cos = np.cos(a)
print(cos)

tan = np.tan(a)
print(tan)

square = np.sqrt(a)
print(square)

y1 = a>= 1
print("a >=1: ", y1)

y2 = a<= 1
print("a <=1: ", y2)

import numpy as np
A = np.array([[1,2,],[4,5]])
B = np.array([[1,2,][4,5]])

Matmul = A @ B
print()


