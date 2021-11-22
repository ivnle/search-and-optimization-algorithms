# %%
import numpy as np
import matplotlib.pyplot as plt


def cross_entropy_methods(f, x0, n_samples, elite_pct, max_iterations):
    """
    Args:
        f: function to minimize
        x0: initial x value
        n_samples: number of samples to sample from gaussian distribution
        elite_pct: % of samples to consider elite
        max_iterations: number of iterations of cross entropy method to perform

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

    # Set initial mean and covariance matrix
    mu = x0
    cvm = np.eye(mu.size)
    n_elite = int(elite_pct * n_samples)

    while True:   
        iterations += 1      
        
        # Sample from gaussian and sort from low to high f values
        samples = np.random.multivariate_normal(mu, cvm, n_samples)
        samples = np.array(sorted(samples, key=lambda x: f(x)))
        
        # Get the n_elite samples with the lowest f values
        elite_samples = samples[:n_elite]
        
        # Update mu and cvm
        mu = np.mean(elite_samples, axis=0)
        
        cvm = np.zeros((mu.size, mu.size))
        for i in range(n_elite):       
            s = elite_samples[i] - mu
            cvm_i = np.outer(s, s)
            cvm += cvm_i
        cvm /= n_elite
        
        x_hist.append(mu)
        f_hist.append(f(mu))

        if iterations > max_iterations:
            break

    return x_hist, f_hist, iterations


if __name__ == '__main__':

    def f(x: np.ndarray) -> float:
        """
        """
        Q = np.array([[2, 0, 0],
                      [0, 4, 0],
                      [0, 0, 6]])
        b = np.array([2, 2, 2])
        return 0.5 * (x.T @ Q @ x) - (b.T @ x)

    
    # random seed
    np.random.seed(8)

    x0 = x_init = np.full(3, 2.)
    n_samples = 200
    elite_pct = 0.1
    iterations = 10

    x_hist, f_hist, iterations = cross_entropy_methods(f, x0, n_samples, elite_pct, iterations)
    print(f'final x value: {x_hist[-1]}')
    print(f'final f value: {f_hist[-1]}')
    print(f'iterations: {iterations-1}')

    # plot f_hist
    plt.plot(f_hist)
    plt.xlabel('iteration')
    plt.ylabel('$f(x)$')
    plt.title('Cross Entropy Methods')
    plt.show()
