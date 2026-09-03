"""
Defines the optimization functions and methods for the GDV package.

The custom optimization methods are designed to be compatible with SciPy's `minimize` function, 
allowing users to easily switch between built-in and custom optimization algorithms.

This is done by passing any `gdv.optimize` functions as the `method` argument to `scipy.optimize.minimize`.

"""
from scipy.optimize import minimize, OptimizeResult
import numpy as np
from typing import Callable, Any
########################################
# Example of a custom method:
########################################


def dumb_optimize(
        fun, 
        x0, 
        args=(), 
        jac: Any | None = None,
        hess: Any | None = None,
        hessp: Any | None = None,
        bounds: Any | None = None,
        constraints: Any | None = None,
        tol: float | None = 1e-8, # tolerance, like our epsilon that defines convergence
        callback: Callable = None, 
        **kwargs
    ) -> OptimizeResult:
    """
    Dummy function to demonstrate how to implement a custom optimization method compatible with SciPy's `minimize`.
    Function signature is demanded by scipy.optimize.minimize, so we must accept the same parameters.

    Parameters:
    - fun: The objective function to be minimized.
    - x0: Initial guess for the parameters.
    - args: Extra arguments passed to the objective function.
    - jac: Jacobian (gradient) of the objective function.
    - hess: Hessian (second derivative) of the objective function.
    - hessp: Hessian product function.
    - bounds: Bounds for variables (only for constrained optimization).
    - constraints: Constraints definition (only for constrained optimization).
    - tol: Tolerance for termination / convergence.
    - callback: A function called after each iteration of the optimization.
    - **kwargs: Additional keyword arguments for custom parameters.

    The `kwargs` can include custom parameters such as:
    - lr: Learning rate for the optimization algorithm.
    - max_iter: Maximum number of iterations for the optimization algorithm.

    Returns:
    - res: An OptimizeResult object containing the optimization results.

    Example usage:

    ```python

        custom_options = {'lr': 0.05, 'max_iter': 25}

        result_custom = minimize(
            fun=objective_func, 
            x0=start_point,
            callback=callback, # assumes you've defined your own callback function to track the optimization history          
            method=dumb_optimize, # custom function
            options=custom_options #  <-- custom params passed here
        )
    ```

    """
    # SciPy bundles the options dict into kwargs.
    # So, minimize(..., **options) ---> custom_gradient_descent(..., **kwargs)
    #########################################
    # Unpack kwargs to get custom parameters
    #########################################

    learning_rate = kwargs.get('lr', 0.1)
    max_iter = kwargs.get('max_iter', 10)
    
    x = np.array(x0)
    step_size = learning_rate

    # run a dummy loop to show the options in action
    for _ in range(max_iter):

        x = x - step_size       # dumb update step
        if step_size < tol:     # arbitrary convergence criterion
            break

        if callback:
            # traditional SciPy callback passes just the current vector `x`
            callback(x)

    # create the mandatory SciPy output wrapper
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