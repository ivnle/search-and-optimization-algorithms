# %%
import numpy as np
import matplotlib.pyplot as plt
import gym
import matplotlib.pyplot as plt
import seaborn as sns

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):

    value_function = np.zeros(nS)

    loops = 0
    while True:
        loops += 1
        value_change = 0
        for state in range(nS):
            action = policy[state]
            old_value = value_function[state]
            new_value = 0
            for prob, next_state, reward, terminal in P[state][action]:
                new_value += prob * \
                    (reward + gamma * value_function[next_state])
            value_function[state] = new_value
            value_change = max(value_change, abs(old_value - new_value))
        if value_change < tol:
            break

    ############################
    return value_function, loops


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):

    new_policy = np.zeros(nS, dtype='int')

    for state in range(nS):
        new_action_value = -float('inf')
        new_action = None

        for action in range(nA):
            value = 0
            for prob, next_state, reward, terminal in P[state][action]:
                value += prob * (reward + gamma *
                                 value_from_policy[next_state])
            if value > new_action_value:
                new_action_value = value
                new_action = action

        assert new_action is not None
        new_policy[state] = new_action

    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3, printout=True):

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    policy_improv_steps = 0
    policy_eval_steps = 0
    while True:
        old_policy = policy.copy()
        value_function, add_policy_eval_steps = policy_evaluation(
            P, nS, nA, policy, gamma, tol)
        policy_eval_steps += add_policy_eval_steps
        policy = policy_improvement(P, nS, nA, value_function, policy, gamma)
        policy_improv_steps += 1
        if np.all(policy == old_policy):
            break

    if printout:
        print(f'{policy_improv_steps} policy improvement steps.')
        print(f'Total policy evaluation steps: {policy_eval_steps}')

    return value_function, policy


def heatmap_value_func(value_func):
    sns.set()
    sns.heatmap(value_func)
    plt.show()


if __name__ == '__main__':

    environment = "FrozenLake8x8-v1"
    env = gym.make(environment)

    gamma = 0.9
    tolerance = 1e-3

    value_function, policy = policy_iteration(
        env.env.P, env.env.nS, env.env.nA, gamma, tolerance)

    print('optimal value function', value_function)
    print('optimal policy', policy)

    # change shape of value_function to 16,1
    value_function = np.reshape(value_function, (8, 8))

    heatmap_value_func(value_function)
