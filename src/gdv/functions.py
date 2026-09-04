"""
Defines example functions for optimization and visualization.

All function variables are defined as elements of tuples.

For example, a function f(x, y) will have x[0] = x and x[1] = y

Also includes the Jacobian and Hessian for each function.
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

########################
# Convex bowl function
########################
# f(x, y) = x^2 + y^2
convex_bowl = lambda x: x[0]**2 + x[1]**2

# f(x,y)/dx = 2x, f(x,y)/dy = 2y
convex_bowl_jacobian = lambda x: np.array([2 * x[0], 2 * x[1]])

# f(x,y)/dx^2 = 2, f(x,y)/dy^2 = 2, f(x,y)/dxdy = 0
# Note: we have a symmetric square matrix, and the hessian happens to be constant..
convex_bowl_hess = lambda x: np.array(
    [
        [2, 0], 
        [0, 2]
    ]
)


########################
# rosenbrock function
########################
#f(x, y) = (1 - x)^2 + 100 (y - x^2)^2
rosenbrock = lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


rosenbrock_jacobian = lambda x: np.array(
    [
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
        200 * (x[1] - x[0]**2)
    ]
)

rosenbrock_hess = lambda x: np.array(
    [
        [1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]],
        [-400 * x[0], 200]
    ]
)

########################
# multimodal_nonconvex
########################
#f(x, y) = x^2 + y^2 + 10 cos(x) + 10 cos(y)
multimodal_nonconvex = lambda x: x[0]**2 + x[1]**2 + 10 * np.cos(x[0]) + 10 * np.cos(x[1])

multimodal_nonconvex_jacobian = lambda x: np.array(
    [
        2*x[0] - 10 * np.sin(x[0]),
        2*x[1] - 10 * np.sin(x[1])
    ]
)

multimodal_nonconvex_hess = lambda x: np.array(
    [
        [2 - 10 * np.cos(x[0]), 0],
        [0, 2 - 10 * np.cos(x[1])]
    ]
)   