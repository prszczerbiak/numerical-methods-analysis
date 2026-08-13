import numpy as np

X = np.linspace(0, 2, 21)
Y = np.linspace(0, 1, 11)

with open('data.txt', 'w') as f:
    for y in Y:
        for x in X:
            z = 900 * np.sin(8 * x) * np.cos(3 * y) - 100
            
            f.write(f"{x:.4f} {y:.4f} {z:.4f}\n")

print("Success")