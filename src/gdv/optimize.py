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


def gradient_descent(
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
    Gradient descent function, as discussed in the week 2 lecture.
    It only needs the Jacobian (gradient) of the objective function, which is passed as `jac` to this function.
    The Jacobian will need to be predefined as a separate function.

    The `kwargs` includes the custom parameter:
    - lr: Learning rate for the optimization algorithm.
    - max_iter: Maximum number of iterations for the optimization algorithm.


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



    Returns:
    - res: An OptimizeResult object containing the optimization results.

    Example usage:

    ```python

        custom_options = {'lr': 0.05, 'max_iter': 50_000}

        result_custom = minimize(
            fun=objective_func, 
            x0=start_point,
            callback=callback, # assumes you've defined your own callback function to track the optimization history          
            method=gradient_descent, # custom function
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
    

    iter_cnt = 0
    # run a dummy loop to show the options in action
    for iter_cnt in range(max_iter):

        step_size = learning_rate * jac(x, *args)  # compute the step size using the Jacobian

        xk = x - step_size  # update the current point
        
        # debugging output to show the optimization process
        # print(f"Iteration {iter_cnt}: x = {x}, \
        #     step_size = {step_size}, \
        #     norm = {np.linalg.norm(xk - x)}"
        # )

        if np.linalg.norm(xk - x) < tol:  # convergence criterion
            print(f"Converged after {iter_cnt} iterations:\nxk = {xk}\nx = {x}\nnorm = {np.linalg.norm(xk - x)}\ntol = {tol}\n")
            break

        if callback:
            # traditional SciPy callback passes just the current vector `x`
            callback(x)

        x = xk # update current to new point

        
    # create the mandatory SciPy output wrapper
    res = OptimizeResult(
        x=x,
        success=True,
        status=0,
        message=f"Custom optimization converged after {iter_cnt} iterations.",
        fun=fun(x, *args),
        nit=iter_cnt,
        nfev=iter_cnt
    )

    return res

def newton(
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
    Newton method, which requires an invertible hessian.

    The `kwargs` includes the custom parameter:
    - max_iter: Maximum number of iterations for the optimization algorithm.


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



    Returns:
    - res: An OptimizeResult object containing the optimization results.

    Example usage:

    ```python

        custom_options = {'lr': 0.05, 'max_iter': 50_000}

        result_custom = minimize(
            fun=objective_func, 
            x0=start_point,
            callback=callback, # assumes you've defined your own callback function to track the optimization history          
            method=gradient_descent, # custom function
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
    

    iter_cnt = 0
    # run a dummy loop to show the options in action
    for iter_cnt in range(max_iter):

        jac_k = learning_rate * jac(x, *args)  # compute the step size using the Jacobian

        hess_k = hess(x, *args)  # compute the Hessian at the current point

        try:
            hess_inv_k = np.linalg.inv(hess_k)  # compute the inverse of the Hessian
            # print(hess_inv_k)
        except np.linalg.LinAlgError:
            print(f"Hessian is not invertible at iteration {iter_cnt}. Stopping optimization.")
            break

        # definition of step size t; not fixed.
        step_size = hess_inv_k @ jac_k  

        xk1 = x - step_size  # update the current point
        
        if np.linalg.norm(xk1 - x) < tol:  # convergence criterion
            print(f"Converged after {iter_cnt} iterations:\nxk = {xk1}\nx = {x}\nnorm = {np.linalg.norm(xk1 - x)}\ntol = {tol}\n")
            break

        if callback:
            # traditional SciPy callback passes just the current vector `x`
            callback(x)

        x = xk1 # update current to new point

        
    # create the mandatory SciPy output wrapper
    res = OptimizeResult(
        x=x,
        success=True,
        status=0,
        message=f"Custom optimization converged after {iter_cnt} iterations.",
        fun=fun(x, *args),
        nit=iter_cnt,
        nfev=iter_cnt
    )

    return res