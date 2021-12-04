# %%
import numpy as np
import matplotlib.pyplot as plt
import gym
import time

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


def value_iteration(P, nS, nA, gamma, tolerance):
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    iterations = 0

    # find optimal value function
    while True:
        iterations += 1
        value_change = 0
        for state in range(nS):
            old_value = value_function[state]
            new_value = -float('inf')
            for action in range(nA):
                current_value = 0
                for prob, next_state, reward, terminal in P[state][action]:
                    current_value += prob * \
                        (reward + gamma * value_function[next_state])
                if current_value > new_value:
                    new_value = current_value
            value_function[state] = new_value
            value_change = max(value_change, abs(old_value - new_value))
        if value_change < tolerance:
            break

    print(
        f'Value iteration required {iterations} iterations to find the optimal value function.')

    # extract policy from optimal value function
    for state in range(nS):
        state_action_values = np.zeros(nA)
        for action in range(nA):
            current_value = 0
            for prob, next_state, reward, terminal in P[state][action]:
                current_value += prob * \
                    (reward + gamma * value_function[next_state])
            state_action_values[action] = current_value
        policy[state] = np.argmax(state_action_values)

    return value_function, policy


def heatmap_value_func(value_func):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    sns.heatmap(value_func)
    plt.show()


if __name__ == '__main__':

    environment = "FrozenLake8x8-v0"
    env = gym.make(environment)

    gamma = 0.9
    tolerance = 1e-3

    value_function, policy = value_iteration(
        env.env.P, env.env.nS, env.env.nA, gamma, tolerance)

    print('optimal value function', value_function)
    print('optimal policy', policy)

    # change shape of value_function to 16,1
    value_function = np.reshape(value_function, (8, 8))

    heatmap_value_func(value_function)

