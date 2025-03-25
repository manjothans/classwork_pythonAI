import numpy as np
z = np.zeros(5)
print(z)
np.shape(z)
z2 = np.zeros((4,5))
print(z2)
np.shape(z2)
y = np.ones((2,3))
print(y)
F= np.full((7,8),11)
print(F)
x = np.linspace(0,5,10)
print(x)
x2 = np.arange(0,5,0.2)
print(x2)

a = 1
b = 2
amount = 50
nopat = np.random.randint(a,b+1,amount)
print(nopat)

x = np.random.randn(100)
print(x)

y = np.random.random(10)
print(y)