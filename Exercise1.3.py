import numpy as np
import matplotlib.pyplot as plt

height = np.array([60, 65, 70, 72, 68])  #  in inches
weight = np.array([150, 170, 180, 200, 160])  #  in pounds

height_cm = height * 2.54 #inches to cm
weight_kg = weight * 0.453592 #pounds to kilo

meanheight = np.mean(height_cm)
meanweight = np.mean(weight_kg)

print("mean Height (cm):", meanheight)
print("mean Weight (kg):",meanweight)

print("converted Data:")
print(f"heights (in cm):" ,height_cm)
print(f"weights (in kg):", weight_kg)

plt.plot(height_cm, weight_kg ,color='blue', marker='o', linestyle='-', label='Height (cm) Histogram')
plt.xlabel('height (cm)')
plt.ylabel('frequency')
plt.title('histogram of Heights in Cm')
plt.grid()
plt.show()