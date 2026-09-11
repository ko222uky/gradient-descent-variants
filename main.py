"""
This main script is for executing all gradient descent variants,
under different hyperparameters, to obtain the raw results for
the project 1 deliverables. 

For runtime statistics, the optimization algorithm was executed N=100 times.
"""
from statistics import mean, stdev
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
from time import perf_counter



def main():

    def callback(x):
        """
        Callback function to log the optimizer's history at each step
        """
        history.append(np.copy(x)) 

    N = 100; # to get runtime averages

    ###############################################################
    # define your method & params function & method
    ###############################################################
    methods = ["gradient_descent", "newton", "adagrad", "adam"]
    
    func_dict = {
        "gradient_descent" : gradient_descent,
        "newton" : newton,
        "adagrad" : adagrad,
        "adam" : adam
    }

    for method in methods:

        df_data = []

        print(f"RUNNING METHOD: {method}")
        my_optimization_method = func_dict[method]

        custom_options_set = [
            {'max_iter': 5000, 'lr' : 0.001},
            {'max_iter': 5000, 'lr' : 0.01},
            {'max_iter': 5000, 'lr' : 0.1},
        ] if method != "newton" else [
            {'max_iter': 5000, 'decay_rate' : 0.000001},
            {'max_iter': 5000, 'decay_rate' : 0.00001},
            {'max_iter': 5000, 'decay_rate' : 0.0}]

        start_points = [
            np.array([-7.0, -4.0]),
            np.array([10, 10]),
            np.array([1, -1])
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



                    # 3. get your X, Y, Z
                    X, Y = default_meshgrid(x_range=(-15, 15), y_range=(-15, 15), num_points=5000)
                    Z = objective_func((X, Y))

                    # 4. run optimization; this is where we get the time.

                    # to get average runtimes + stdev, I will run this N=10 times.

                    runtimes = []
                    result = {}

          

                    print(f"Performing {N} runs to get runtime stats.")
                    for _ in range(N):

                        history = []
                        history.append(start_point) 

                        # define callback

                        start = perf_counter()
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
                        end = perf_counter()
                        runtimes.append(end - start)

                    runtime_dict = {
                        'avg_runtime' : mean(runtimes),
                        'stdev_runtime' : stdev(runtimes),
                        'N_runs' : N
                    }

                    print(runtime_dict)

                    data_row |= result
                    data_row |= runtime_dict

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

                    custom_option = custom_options.get('decay_rate', None) if method == "newton" else custom_options.get('lr', None) 

                    fig.savefig(Path(f'figures/{method}/{obj_func}_{custom_option}_{start_point[0]}_{start_point[1]}.png'), dpi=300)

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