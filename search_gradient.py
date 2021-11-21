# %%
import numpy as np
import matplotlib.pyplot as plt

def grad_gauss_wrt_mu(z, mu, inv_cvm):
    return inv_cvm @ (z - mu)

def grad_gauss_wrt_cvm(z, mu, inv_cvm):
    normalized_z = z - mu
    return -0.5 * (inv_cvm - (inv_cvm @ np.outer(normalized_z, normalized_z) @ inv_cvm))

def search_gradient(f, x0, n_samples, max_iterations, step_size):
    """
    Args:
        f: function to minimize
        x0: initial x value
        n_samples: number of samples to sample from gaussian distribution
        max_iterations: number of iterations of cross entropy method to perform
        step_size: step size for parameter updates

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

    while True:   
        iterations += 1      
        inv_cvm = np.linalg.inv(cvm)
        samples = np.random.multivariate_normal(mu, cvm, n_samples)
        sample_means = np.array([grad_gauss_wrt_mu(z, mu, inv_cvm)*f(z) for z in samples])
        sample_cvms = np.array([grad_gauss_wrt_cvm(z, mu, inv_cvm)*f(z) for z in samples])
        grad_J_wrt_mu = np.mean(sample_means, axis=0)
        grad_J_wrt_cvm = np.mean(sample_cvms, axis=0)
        
        mu -= step_size * grad_J_wrt_mu
        cvm -= step_size * grad_J_wrt_cvm        
        
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
    n_samples = 100
    iterations = 500
    step_size = 0.001

    x_hist, f_hist, iterations = search_gradient(f, x0, n_samples, iterations, step_size)

    x_hist = np.array(x_hist)
    f_hist = np.array(f_hist)
    min_idx = np.argmin(f_hist)

    print(f'min x value: {x_hist[min_idx]}')
    print(f'min f value: {f_hist[min_idx]}')
    print(f'iterations: {iterations-1}')
    

    # plot f_hist
    plt.plot(f_hist)
    plt.xlabel('iteration')
    plt.ylabel('$f(x)$')
    plt.show()
