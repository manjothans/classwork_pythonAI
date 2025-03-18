import math
from operator import floordiv

a = int(input("Enter the first number (a): "))
b = int(input("Enter the second number (b): "))

# Perform calculations
add = a + b
sub = a - b
multi = a * b
div = a / b
floordiv = a//b
per = a % b
expo = a ** b

# Display the results
print(f"Addition of  {a} + {b} is {add}")
print(f"Substraction of  {a} - {b} is {sub}")
print(f"Multiplication of  {a} * {b} is {multi}")
print(f"Division of  {a} / {b} is {div}")
print(f"Floor division of  {a} // {b} is {floordiv}")
print(f"Modulus of {a} % {b} is {per}")
print(f"Exponentiation of {a} ** {b} is {expo}")