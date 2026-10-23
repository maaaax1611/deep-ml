import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    # edge cases
    # so we are gonna use centered differences here
    x = np.asarray(x, dtype=float)
    numerical_grad = np.zeros_like(x)

    for i in range(x.size):
        e = np.zeros_like(x)
        e.flat[i] = epsilon
        numerical_grad.flat[i] = (f(x+e)-f(x-e)) / (2*epsilon)

    denom = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)
    relative_error = 0.0 if denom == 0 else np.linalg.norm(numerical_grad - analytical_grad) / denom
    
    return (numerical_grad, relative_error)
