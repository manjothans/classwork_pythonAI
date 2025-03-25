import numpy as np
a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
reshaped = np.reshape(a, (3,4))
rows, cols = np.shape(reshaped)
print(rows, cols)

New1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
new2 = New1.reshape(4,3)
print(new2)
print(new2[0,:])