# %%
import numpy as np
import matplotlib.pyplot as plt

def calc_step_size(gf, Q, x, p):
    num = gf(x) @ p
    den = p @ Q @ p
    return -num/den

def calc_r(gf, x):
    return gf(x)

def calc_beta(p, Q, r):
    num = p @ Q @ r
    den = p @ Q @ p
    return num / den

def calc_next_direction(gf, Q, x, p):
    r = calc_r(gf, x)
    beta = calc_beta(p, Q, r)
    return -r + beta * p

def conjugate_descent(f, gf, hf, x0, tolerance):
    """
    Args:
        f: quadratic function to minimize
        gf: gradient of f
        hf: hessian of f
        x0: initial x value
        step_size: step size
        tolerance: tolerance for stopping condition

    Returns:
        x_hist: history of x values as gradient descent updates x
        f_hist: history of f(x) values as gradient descent updates x
        iterations: number of iterations to reach tolerance
    """

    x_hist = []
    x_hist.append(x0)

    f_hist = []
    f_hist.append(f(x0))

    iterations = 0

    x = x0    
    Q = hf(x)
    p = gf(x)
    step_size = calc_step_size(gf, Q, x, p)

    x += step_size * p

    while True:                
        p = calc_next_direction(gf, Q, x, p)
        step_size = calc_step_size(gf, Q, x, p)
        x += step_size * p
        
        x_hist.append(x)
        f_hist.append(f(x))
        iterations += 1

        print('p:', p)
        if np.linalg.norm(p) < tolerance:
            break

    return x_hist, f_hist, iterations


if __name__ == '__main__':

    def f(x: np.ndarray) -> float:
        """
        f(x) = x_1**2 + x_2**2 + x_3**2 ... + x_n**2
        """
        Q = np.array([[2, 0, 0],
                      [0, 4, 0],
                      [0, 0, 6]])
        b = np.array([2, 2, 2])
        return 0.5 * (x.T @ Q @ x) - (b.T @ x)

    def gf(x: np.ndarray) -> np.ndarray:
        """
        """
        Q = np.array([[2, 0, 0],
                      [0, 4, 0],
                      [0, 0, 6]])
        b = np.array([2, 2, 2])
        return Q @ x - b

    def hf(x: np.ndarray) -> np.ndarray:
        """
        """
        Q = np.array([[2, 0, 0],
                      [0, 4, 0],
                      [0, 0, 6]])
        return Q

    x0 = np.array([2., -2., 2])
    tolerance = 1e-5

    x_hist, f_hist, iterations = conjugate_descent(
        f, gf, hf, x0, tolerance)
    print(f'final x value: {x_hist[-1]}')
    print(f'final f value: {f_hist[-1]}')
    print(f'iterations: {iterations-1}')

    # plot f_hist
    plt.plot(f_hist)
    plt.xlabel('iteration')
    plt.ylabel('$f(x)$')
    plt.show()
