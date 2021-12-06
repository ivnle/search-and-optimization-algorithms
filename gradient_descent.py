# %%
import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(f, gf, x0, step_size, tolerance):
    """
    Args:
        f: function to minimize
        gf: gradient of f
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

    while True:
        x += step_size * -gf(x)

        x_hist.append(x)
        f_hist.append(f(x))
        iterations += 1

        if abs(f_hist[-1] - f_hist[-2]) < tolerance:
            break

    return x_hist, f_hist, iterations


if __name__ == '__main__':

    def f(x: np.ndarray) -> float:
        """
        f(x) = x_1**2 + x_2**2 + x_3**2 ... + x_n**2
        """
        return np.linalg.norm(x)**2

    def gf(x: np.ndarray) -> np.ndarray:
        """
        gf(x) = 2 * x
        """
        return 2 * x

    x0 = np.array([2., -2., 2])

    step_size = 0.01
    tolerance = 1e-5

    x_hist, f_hist, iterations = gradient_descent(
        f, gf, x0, step_size, tolerance)
    print(f'min x value: {x_hist[-1]}')
    print(f'min f value: {f_hist[-1]}')
    print(f'iterations to find minimum (within tolerance): {iterations-1}')

    # plot f_hist
    plt.plot(f_hist)
    plt.xlabel('iteration')
    plt.ylabel('$f(x)$')
    plt.title('Gradient descent')
    plt.show()
