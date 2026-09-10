"""
This main script is for testing the functionality of the package implementations.

For the Project 1 CSE620 deliverables, the following must be performed:

What to Implement / Do:
A. Implement or use library versions of the four optimizers (NumPy, PyTorch, JAX, etc.).

B. For each function, pick 2–3 different initial points (e.g., (-2, 2), (0.5, -1.5), (3, 3)).

C. For each optimizer, try at least 3 step sizes (also called learning rates) e.g., 0.001, 0.01, 0.1. Tune
Newton’s with damping (damping means decreasing rate gradually with each iteration) if needed.

D. Run until convergence, i.e., ||x_(k+1) - x_k|| < 1e-6. (However, you may need to limit the number of
iterations to say, 2000 in case of either very slow convergence or oscillation)

E. Visualize the process by drawing contour plots and overlaying the optimization path.

F. Record the number of iterations to converge, final point, and the final f value.

G. Analyze your results by discussing the convergence behavior across functions and
hyperparameters. Identify when and why methods differ.

Expected Phenomena to Observe:
● f1 (quadratic): Newton converges in one step from generic starts; GD converges smoothly if α
is reasonable; AdaGrad/Adam also converge quickly.

● f2 (Rosenbrock): GD often “zigzags” or diverges unless α is very small; AdaGrad may slow
down; Adam typically works well with moderate α.

● f3 (cosine bumps): Multiple local minima; different methods and α may land in different basins;
Adam’s momentum may escape shallow minima better than plain GD/AdaGrad

"""
import gc
from pathlib import Path
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
from gdv.optimize import (
    gradient_descent,
    newton,
    adagrad,
    adam
)
import polars as pl

def main():

    df_data = []

    ###############################################################
    # define your method & params function & method
    ###############################################################

    # method = "gradient_descent"
    # method = "newton"
    method = "adagrad"
    # method = "adam"


    func_dict = {
        "gradient_descent" : gradient_descent,
        "newton" : newton,
        "adagrad" : adagrad,
        "adam" : adam
    }

    my_optimization_method = func_dict[method]

    custom_options_set = [
        # {'max_iter': 5000, 'decay_rate' : 0.000001},
        # {'max_iter': 5000, 'decay_rate' : 0.00001},
        # {'max_iter': 5000, 'decay_rate' : 0.0},

        {'max_iter': 5000, 'lr' : 0.001},
        {'max_iter': 5000, 'lr' : 0.01},
        {'max_iter': 5000, 'lr' : 0.1},
    ]

    start_points = [
        np.array([-7.0, -4.0]),
        np.array([10, 10])
    ]

    
    ###############################################################
    # define your objective function (incl. jacobian and hessian)
    ###############################################################
    obj_func_dict = {
        "nonconvex": (multimodal_nonconvex, multimodal_nonconvex_jacobian, multimodal_nonconvex_hess),
        "rosenbrock":(rosenbrock, rosenbrock_jacobian, rosenbrock_hess),
        "convex_bowl": (convex_bowl, convex_bowl_jacobian, convex_bowl_hess) 
    }

    obj_funcs = ['nonconvex', 'rosenbrock', 'convex_bowl']


    for obj_func in obj_funcs:

        objective_func = obj_func_dict[obj_func][0]
        objective_jacobian = obj_func_dict[obj_func][1]
        objective_hessian = obj_func_dict[obj_func][2]  # Not used in this example, but can be defined if needed

        # 1. define your starting point and initialize the history list
        for custom_options in custom_options_set:
            for start_point in start_points:

                data_row = {}

                # add obj func name
                data_row |= {"Objective Function" : obj_func}


                # add start point
                data_row |= {
                    'Start X' : float(start_point[0]),
                    'Start Y' : float(start_point[1])
                }
                data_row |= custom_options

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
                    tol=1e-6 
                )
                data_row |= result

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

                fig.savefig(Path(f'figures/{method}/{obj_func}_{custom_options}_{start_point}.png'), dpi=300)

                # recast and format some of the results so it can be written to CSV...
                data_row['fun'] = float(data_row['fun'])
                data_row['x'] = str(data_row['x'][0]) + ', ' + str(data_row['x'][1])

                print(data_row)

                df_data.append(data_row)

                gc.collect()
                plt.close(fig)  # close the figure after saving, else memory leak

    df = pl.DataFrame(data=df_data)
    print(df)
    df.write_csv(Path(f'figures/{method}/data.csv'))

if __name__ == '__main__':
    main()