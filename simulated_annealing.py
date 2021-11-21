# %%
import numpy as np
import matplotlib.pyplot as plt

def fast_annealing_schedule(T, i):
    return T / i

def exponential_annealing_schedule(T, gamma):
    """
    Args:
        T: current temperature
        gamma: in (0, 1)
    """
    return T * gamma

def log_annealing_schedule(T, i):
    num = T * np.log(2)
    den = np.log(i+1)
    return num / den

def simulated_annealing(f, x0, T, tolerance):
    """
    Args:
        f: function to minimize
        x0: initial x value
        T: initial temperature
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

    while True:   
        iterations += 1      
        T = log_annealing_schedule(T, iterations)

        # generate new sample       
        x_new = np.random.multivariate_normal(x, np.identity(len(x)))

         # accept new sample if it is better
        if f(x_new) < f(x):
            x = x_new
        # otherwise, accept with some probability
        else:
            if np.random.rand() < np.exp((f(x) - f(x_new)) / T):
                x = x_new        
        
        
        x_hist.append(x)
        f_hist.append(f(x))

        # if f_hist[-1] - f_hist[-2] < tolerance:
        #     break
        if iterations > 1000:
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

    
    # random seed
    np.random.seed(8)

    x0 = x_init = np.full(3, 2.)
    tolerance = 1e-5
    T = 100

    x_hist, f_hist, iterations = simulated_annealing(f, x0, T, tolerance)
    print(f'final x value: {x_hist[-1]}')
    print(f'final f value: {f_hist[-1]}')
    print(f'iterations: {iterations-1}')

    # plot f_hist
    plt.plot(f_hist)
    plt.xlabel('iteration')
    plt.ylabel('$f(x)$')
    plt.show()
