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
    max_iter = kwargs.get('max_iter', 2000)
    
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
            # print(f"Converged after {iter_cnt} iterations:\nxk = {xk}\nx = {x}\nnorm = {np.linalg.norm(xk - x)}\ntol = {tol}\n") # DEBUG
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
        nit=iter_cnt+1,
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
        tol: float | None = 1e-6, # tolerance, like our epsilon that defines convergence
        callback: Callable = None, 
        **kwargs
    ) -> OptimizeResult:
    """
    Newton method, which requires an invertible hessian.

    The `kwargs` includes the custom parameter:
    - max_iter: Maximum number of iterations for the optimization algorithm.
    - decay_rate: Deterministic decay rate that is linear. Default is 0.1.
    
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

    learning_rate = kwargs.get('lr', 1.0)
    max_iter = kwargs.get('max_iter', 2000)
    decay_rate = kwargs.get('decay_rate', 0.1)

    ############################
    # initializations
    ############################

    x = np.array(x0)

    iter_cnt = 0

    decay = 1

    inverse_time_decay = learning_rate


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

        
        inverse_time_decay /= decay

        # definition of step size t; not fixed.
        step_size = hess_inv_k @ jac_k

        xk = x - inverse_time_decay*step_size  # update the current point

        if callback:
            # log the new point (xk), not the stale x, so the final
            # converged point is always captured in the history.
            callback(xk)

        if np.linalg.norm(xk - x) < tol:  # convergence criterion
            #print(f"Converged after {iter_cnt} iterations:\nxk = {xk}\nx = {x}\nnorm = {np.linalg.norm(xk - x)}\ntol = {tol}\n")
            x = xk
            break

        x = xk
        decay += decay_rate

    # create the mandatory SciPy output wrapper
    res = OptimizeResult(
        x=x,
        success=True,
        status=0,
        message=f"Custom optimization converged after {iter_cnt} iterations.",
        fun=fun(x, *args),
        nit=iter_cnt+1,
        nfev=iter_cnt
    )

    return res

def adam(
        fun, 
        x0, 
        args=(),
        jac: Any | None = None,
        hess: Any | None = None,
        hessp: Any | None = None,
        bounds: Any | None = None,
        constraints: Any | None = None,
        tol: float | None = 1e-8, # tolerance isn't needed for ada_grad; convergence guaranteed
        callback: Callable = None, 
        **kwargs
    ) -> OptimizeResult:
    """
    Adaptive Moment Estimation (ADAM) combines RMSprop and momentum.
    Momentum keeps an exponentially decaying average of gradients, whereas
    RMSprop keeps an exponentially decaying average of squared gradients.

    Mathematically, this is defined as:

        mk+1 = β1mk + (1-β1)∇f(xk) (momentum ~ mean)
        vk+1 = β2vk + (1-β2)∇f(xk)2 (RMSprop ~ variance)
        m̂k = mk / (1- β1k) (bias correction since m0, v0 = 0)
        v̂k = vk / (1 - β2k)
        xk = xk-1 - αm̂k / (√v̂k + ε) 

    The `kwargs` includes the custom parameter:

    - lr: Learning rate for the optimization algorithm. Default 0.01.
    - epsilon: Small constant to prevent division by zero in the AdaGrad update. Default 1e-8.
    - max_iter: Maximum number of iterations for the optimization algorithm. Default 2000.
    - beta1: Scaling factor for momentum. Default 0.9.
    - beta2: Scaling factor for RMSprop. Default 0.999.

    Parameters:
    - fun: The objective function to be minimized.
    - x0: Initial guess for the parameters.
    - args: Extra arguments passed to the objective function.
    - jac: Jacobian (gradient) of the objective function.
    - hess: Hessian (second derivative) of the objective function.
    - hessp: Hessian product function.
    - bounds: Bounds for variables (only for constrained optimization).
    - constraints: Constraints definition (only for constrained optimization).
    - callback: A function called after each iteration of the optimization.
    - **kwargs: Additional keyword arguments for custom parameters.


    Returns:
    - res: An OptimizeResult object containing the optimization results.

    Example usage:

    ```python

        custom_options = {'lr': 0.05, 'epsilon': 1e-8, 'max_iter': 5_000}

        result_custom = minimize(
            fun=objective_func, 
            x0=start_point,         
            callback=callback, # assumes you've defined your own callback function to track the optimization history
            method=ada_grad, # custom function
            options=custom_options #  <-- custom params passed here
        )
    ```

    """

    #########################################
    # Unpack kwargs to get custom parameters
    #########################################

    learning_rate = kwargs.get('lr', 0.01)
    max_iter = kwargs.get('max_iter', 2000)
    epsilon = kwargs.get('epsilon', 1e-8)  # Small constant to prevent division by zero
    beta1 = kwargs.get('beta1', 0.9)
    beta2 = kwargs.get('beta2', 0.999)

    ###################
    # Initializations
    ###################
    x = np.array(x0)    

    m = np.zeros(len(x)) # Initialize the momentum vector where m0 := 0
    v = np.zeros(len(x)) # initialize moment vector where v0 := 0

    iter_cnt = 0

    ###################
    # Main alg loop
    ###################

    for iter_cnt in range(max_iter):
        # computer current gradient and its self-outer-product
        jac_k = jac(x, *args) # compute the Jacobian at the current point 

        # update our momentum and rms prop (i.e., ~ mean & ~ variance)
        m = (beta1 * m) + (1 - beta1) * jac_k     # update momentum
        v = (beta2 * v) + (1 - beta2) * jac_k**2   # update RMSprop; use element-wise squaring
        
        # scaling factors are to the power of t, which is
        # based on the iter count. It starts at t := 1, so add 1.
        m_hat = m / (1 - beta1**(iter_cnt + 1))
        v_hat = v / (1 - beta2**(iter_cnt + 1))

        step_size = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon) 

        xk = x - step_size          # get next point, i.e., take our step

        if np.linalg.norm(xk - x) < tol:  # convergence criterion
            print(f"Converged after {iter_cnt} iterations:\nxk = {xk}\nx = {x}\nnorm = {np.linalg.norm(xk - x)}\ntol = {tol}\n")
            break
        
        if callback:
            callback(x)

        x = xk

    # print(f"Ended after {iter_cnt + 1} iterations:\nxk = {xk}\nx = {x}\nnorm = {np.linalg.norm(xk - x)}") 

    # create the mandatory SciPy output wrapper
    res = OptimizeResult(
        x=x,
        success=True,
        status=0,
        message=f"Custom optimization converged after {iter_cnt} iterations.",
        fun=fun(x, *args),
        nit=iter_cnt+1,
        nfev=iter_cnt
    )

    return res

def adagrad(
        fun, 
        x0, 
        args=(), 
        jac: Any | None = None,
        hess: Any | None = None,
        hessp: Any | None = None,
        bounds: Any | None = None,
        constraints: Any | None = None,
        tol: float | None = 1e-8, 
        callback: Callable = None, 
        **kwargs
    ) -> OptimizeResult:
    """
    Adaptive gradient implementation, 
    with correction added to avoid singularity problem.
    This particular implementation uses only the diagonal elements of the adjusted learning rate.

    Mathematically, this is defined as:

        x_k+1 = x_k - α diag(εI + Gt)-½ ∇f(xk).

    where α is a fixed learning rate that is scaled by the adaptive gradient matrix G.

    The `kwargs` includes the custom parameter:
    - lr: Learning rate for the optimization algorithm. Default 0.1.
    - epsilon: Small constant to prevent division by zero in the AdaGrad update. Default 1e-8.
    - max_iter: Maximum number of iterations for the optimization algorithm. Default 2000.


    Parameters:
    - fun: The objective function to be minimized.
    - x0: Initial guess for the parameters.
    - args: Extra arguments passed to the objective function.
    - jac: Jacobian (gradient) of the objective function.
    - hess: Hessian (second derivative) of the objective function.
    - hessp: Hessian product function.
    - bounds: Bounds for variables (only for constrained optimization).
    - constraints: Constraints definition (only for constrained optimization).
    - callback: A function called after each iteration of the optimization.
    - **kwargs: Additional keyword arguments for custom parameters.


    Returns:
    - res: An OptimizeResult object containing the optimization results.

    Example usage:

    ```python

        custom_options = {'lr': 0.05, 'epsilon': 1e-8, 'max_iter': 5_000}

        result_custom = minimize(
            fun=objective_func, 
            x0=start_point,         
            callback=callback, # assumes you've defined your own callback function to track the optimization history
            method=ada_grad, # custom function
            options=custom_options #  <-- custom params passed here
        )
    ```

    References:

    * https://optimization.cbe.cornell.edu/index.php?title=AdaGrad

    """

    #########################################
    # Unpack kwargs to get custom parameters
    #########################################

    learning_rate = kwargs.get('lr', 0.1)
    max_iter = kwargs.get('max_iter', 2000)
    epsilon = kwargs.get('epsilon', 1e-8)  # Small constant to prevent division by zero

    x = np.array(x0)        
    G = np.zeros((len(x), len(x))) # Initialize the matrix G as an NxN matrix of zeros

    # print(f"G zero: {G}\n\n") # DEBUG - sanity check for dimensions
    

    iter_cnt = 0


    for iter_cnt in range(max_iter):

        # computer current gradient and its self-outer-product
        jac_k = jac(x, *args) # compute the Jacobian at the current point 

        G += np.outer(jac_k, jac_k)                     # accumulate the squared gradients
        adj_lr = learning_rate / (np.sqrt(G) + epsilon) # adj lr = learning rate divided by G + correction          
        adj_lr = np.diag(adj_lr)                        # note to self: np.diag doesn't return a matrix, but a 1d array.
        
        # DEBUG PRINTS: sanity check
        # print(f"adj lr: {adj_lr}")                                                 # DEBUG
        # if iter_cnt == 1: print(adj_lr, jac_k, adj_lr @ jac_k, x - adj_lr @ jac_k) # DEBUG

        # we thus have two 1d arrays, so dims match and hadamard works
        step_size = adj_lr * jac_k  # elementwise multiplication (hadamard)

        xk = x - step_size          # get next point, i.e., take our step

        if np.linalg.norm(xk - x) < tol:  # convergence criterion
            # print(f"Converged after {iter_cnt} iterations:\nxk = {xk}\nx = {x}\nnorm = {np.linalg.norm(xk - x)}\ntol = {tol}\n")
            break
        
        if callback:
            callback(x)

        x = xk

    print(f"Ended after {iter_cnt + 1} iterations:\nxk = {xk}\nx = {x}\nnorm = {np.linalg.norm(xk - x)}") 

    # create the mandatory SciPy output wrapper
    res = OptimizeResult(
        x=x,
        success=True,
        status=0,
        message=f"Custom optimization converged after {iter_cnt} iterations.",
        fun=fun(x, *args),
        nit=iter_cnt+1,
        nfev=iter_cnt
    )

    return res

