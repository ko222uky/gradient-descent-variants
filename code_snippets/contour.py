"""
This script gives example code on
visualizing the contour of the objective function.

It also shows an example on how a custom method
can be implemented and passed to scipy.optimize.minimize.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, OptimizeResult
from typing import Any

########################################
# Define the objective function 
# (Rosenbrock function).
########################################

# Two ways to define it:
# as a standard function 
# or as a lambda function.

def objective_func(coords):
    x0=coords[0]
    x1=coords[1]
    return (1 - x0)**2 + 100 * (x1 - x0**2)**2

rosenbrock = lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

######################################
# Initialize start & history
######################################

start_point = np.array([-1.5, 2.0])

history = []

history.append(start_point) 


def callback(x: Any):
    """
    Callback function to log the optimizer's history at each step
    """
    history.append(np.copy(x))


########################################
# do your optimization work here
########################################

# minimize can take a callable as a method:
# "The callable is called as 
# method(fun, x0, args, **kwargs, **options) 
# where kwargs corresponds to any other parameters 
# passed to minimize (such as callback, hess, etc.), 
# except the options dict, which has its contents also 
# passed as method parameters pair by pair. 


########################################
# Example of a custom method:
########################################

def custom_gradient_descent(fun, x0, args=(), callback=None, **kwargs):
    # SciPy bundles the options dict into kwargs.
    # So, minimize(..., **options) ---> custom_gradient_descent(..., **kwargs)

    ###############################
    # Unpack kwargs 
    ###############################

    learning_rate = kwargs.get('lr', 0.1)
    max_iter = kwargs.get('max_iter', 10)
    
    x = np.array(x0)
    
    # Run a dummy loop to show the options in action
    for _ in range(max_iter):
        x = x - learning_rate  # Simple update step
        if callback:
            # Traditional SciPy callback style passes just the current vector `x`
            callback(x)

    # Create the mandatory SciPy output wrapper
    res = OptimizeResult(
        x=x,
        success=True,
        status=0,
        message=f"Custom optimization converged after {max_iter} iterations.",
        fun=fun(x, *args),
        nit=max_iter,
        nfev=max_iter
    )

    return res



# you can call minimize and 
# pass your configuration via the 'options' dictionary
custom_options = {'lr': 0.05, 'max_iter': 25}

result_custom = minimize(
    fun=objective_func, 
    x0=start_point,

    # callback=callback,            # comment this out, since the same
                                    # history array is referenced by 
                                    # the callback function.

    method=custom_gradient_descent, # custom function
    options=custom_options          #  <-- custom params passed here
)

print(result_custom)

# You can edit this and switch result_custom with result
# and see that our custom function works. 
# Even if it's not good.

########################################
# Get actual results that we will plot
########################################

# the callback is what captures our path
result = minimize(
    fun=objective_func, 
    x0=start_point, 
    method='Nelder-Mead', 
    callback=callback
)

########################################
# Prepare plot
# for this, we basically evaluate the 
# objective function over a grid, i.e.,
# our search space.
########################################

# Create a grid for the contour plot
x_range = np.linspace(-2, 2, 200)
y_range = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x_range, y_range)


# Evaluate the objective function over the mesh grid.
# since the input expects tuples where index 0 := x and index 1 := y,
# we pack X, Y into an array.
Z = rosenbrock(np.array([X, Y]))

# Plotting
plt.figure(figsize=(10, 7))

#################################################################################
# Draw filled contours with a color map 
# ('jet', 'viridis', or 'coolwarm')
#################################################################################
num_contour_levels = 25

contour_filled = plt.contourf(
    X, Y, Z, 
    cmap='coolwarm',  
    alpha=0.7,
    levels=num_contour_levels
)

plt.colorbar(contour_filled, label='Objective Value')

# draws the distinct contour line boundaries
contours = plt.contour(
    X, Y, Z, 
    levels=num_contour_levels, 
    colors='black', 
    linewidths=0.5
)

# convert history to a numpy array for easy slicing,
# no [][] funny biz
# This represents the actual path we took. 

history = np.array(history)


# overlay the optimization path
plt.plot(
    history[:, 0], # all rows of our X 
    history[:, 1], # all rows of our Y
    color='darkred', 
    marker='o', 
    markersize=4, 
    linestyle='-', 
    linewidth=1.5, 
    label='Optimization Path')

# Highlight Start and Global Optimum (1, 1)
plt.scatter(
    start_point[0], 
    start_point[1], 
    color='red', 
    marker='X', 
    s=150, 
    zorder=5,# make it sit on top of layers
    label='Start'
)


# rosenbrock function's known global optimum is at 1,1
plt.scatter(
    1, 
    1, 
    color='gold', 
    marker='*',  # a gold star!
    s=200, 
    edgecolor='black', 
    zorder=5, 
    label='Optimum (1,1)'
)

# Graph details
plt.title('Optimization Path on Objective Function Contour')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

