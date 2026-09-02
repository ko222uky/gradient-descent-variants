"""
Small working example of np.meshgrid
to understanding the meshgrid output.
"""
import numpy as np

x = np.array([1, 2, 3])
y = np.array([10, 20])

X, Y = np.meshgrid(x, y)    # X, Y defined from returned tuple
mesh = np.meshgrid(x, y)    # whereas mesh is just the whole tuple

print("X matrix:\n", X)

print("\nY matrix:\n", Y)


# Define a dummy mathematical function
def calculate_z(coords):
    xval = coords[0]
    yval = coords[1]
    return (xval ** 2) + yval

# Pass the full meshgrid directly into the function
print("\n\nPassing meshgrid to calculate Z:\n")

Z = calculate_z(mesh)

print(Z)

