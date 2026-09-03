"""
Defines example functions for optimization and visualization.

All function variables are defined as elements of tuples.

For example, a function f(x, y) will have x[0] = x and x[1] = y

"""
import numpy as np


def default_meshgrid(
        x_range=(-4, 4), 
        y_range=(-4, 4), 
        num_points=400):
    """
    Generates a meshgrid for contour plotting.

    Returns:
        X, Y: Meshgrid arrays for the specified ranges and number of points.
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    return X, Y


# f1(x, y) = x^2 + y^2
convex_bowl = lambda x: x[0]**2 + x[1]**2

#f2(x, y) = (1 - x)^2 + 100 (y - x^2)^2
rosenbrock = lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

multimodal_nonconvex = lambda x: x[0]**2 + x[1]**2 + 10 * np.cos(x[0]) + 10 * np.cos(x[1])