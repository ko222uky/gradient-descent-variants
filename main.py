import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from gdv.contour import optim_contour
from gdv.functions import default_meshgrid, rosenbrock, multimodal_nonconvex, convex_bowl

def main():

    # 0. define your objective function
    objective_func = multimodal_nonconvex

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

    # 4. run optimization
    result = minimize(
        fun=objective_func, 
        x0=start_point, 
        method='Nelder-Mead', 
        callback=callback
    )

    # 5. visualize the optimization path on the contour plot
    fig = optim_contour(
        X, Y, Z, 
        history=history, 
        start_point=start_point, 
        #known_optimum=np.array([1.0, 1.0]), 
        figsize=(8, 6), 
    )
    fig.savefig('figures/example.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()