import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from gdv.contour import optim_contour
from gdv.functions import (
    default_meshgrid, 

    multimodal_nonconvex, 
    multimodal_nonconvex_jacobian,
    multimodal_nonconvex_hess,

    rosenbrock,
    rosenbrock_jacobian,
    rosenbrock_hess,

    convex_bowl,
    convex_bowl_jacobian,
    convex_bowl_hess

)
from gdv.optimize import gradient_descent, newton, adagrad

def main():

    ###############################################################
    # a. define your objective function (incl. jacobian and hessian)
    ###############################################################
    objective_func = multimodal_nonconvex
    objective_jacobian = multimodal_nonconvex_jacobian
    objective_hessian = multimodal_nonconvex_hess  # Not used in this example, but can be defined if needed

    ###############################################################
    # b. define your method & params function & method
    ###############################################################
    my_optimization_method = adagrad
    custom_options = {'max_iter': 2000, 'lr' : 0.1}

    ###############################################################
    # c. define your method & params function & method
    ###############################################################
    start_point = np.array([-9.0, -5.0])

    # 1. define your starting point and initialize the history list

    history = []
    history.append(start_point) 

    # 2. define callback
    def callback(x):
        """
        Callback function to log the optimizer's history at each step
        """
        history.append(np.copy(x))

    # 3. get your X, Y, Z
    X, Y = default_meshgrid(x_range=(-15, 15), y_range=(-15, 15), num_points=5000)
    Z = objective_func((X, Y))

    # 4. run optimization

    result = minimize(
        fun=objective_func, 
        jac=objective_jacobian,
        hess=objective_hessian,
        x0=start_point, 
        method=my_optimization_method, 
        callback=callback,
        options=custom_options,
        # tol=1e-5 
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
        end_point=True,
        legend=True,
        figsize=(8, 6), 
    )
    fig.savefig('figures/example.png', dpi=300)
    # plt.show()

if __name__ == '__main__':
    main()