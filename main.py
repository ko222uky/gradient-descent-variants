import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from gdv.contour import optim_contour
from gdv.functions import (
    default_meshgrid, 

    multimodal_nonconvex, 
    multimodal_nonconvex_jacobian,

    rosenbrock,
    rosenbrock_jacobian,

    convex_bowl,
    convex_bowl_jacobian

)
from gdv.optimize import gradient_descent


def main():

    # 0. define your objective function
    objective_func = rosenbrock
    objective_jacobian = rosenbrock_jacobian

    # 1. define your starting point and initialize the history list
    start_point = np.array([-1.5, 2.0])
    history = []
    history.append(start_point) 

    # 2. define callback
    def callback(x):
        """
        Callback function to log the optimizer's history at each step
        """
        history.append(np.copy(x))

    # 3. get your X, Y, Z
    X, Y = default_meshgrid(x_range=(-5, 5), y_range=(-5, 5), num_points=900)
    Z = objective_func((X, Y))

    # 4. define parameters and run optimization

    custom_options = {'lr': 0.001, 'max_iter': 2000}

    result = minimize(
        fun=objective_func, 
        jac=objective_jacobian,
        x0=start_point, 
        method=gradient_descent, 
        callback=callback,
        options=custom_options,
        tol=1e-10 
    )

    print("Optimization Result:")
    print(f"Final point: {result.x}")
    print(f"Function value at final point: {result.fun}")
    print(f"Number of iterations: {result.nit}")
    

    # 5. visualize the optimization path on the contour plot
    fig = optim_contour(
        X, Y, Z, 
        history=history, 
        start_point=start_point, 
        #known_optimum=np.array([1.0, 1.0]), 
        figsize=(8, 6), 
    )
    fig.savefig('figures/example.png', dpi=300)
    # plt.show()

if __name__ == '__main__':
    main()