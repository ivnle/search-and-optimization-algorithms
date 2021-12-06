# %%
import gym
from policy_iteration import policy_iteration, heatmap_value_func
import numpy as np
from collections import defaultdict


def reward_to_go(state, k, episode, gamma):
    reward = 0
    for i, (s, r) in enumerate(episode[k:]):
        reward += (gamma ** i) * r
    return reward


def td_policy_eval(policy, env, gamma, alpha):
    """
    Goal: Estimate value of a policy without knowing the transition probabilities.
    """
    
    # Value function we are trying to approximate
    U = np.zeros(env.nS)

    iterations = 100000
    ob = env.reset()

    for i in range(iterations):
        a = policy[ob]
        new_ob, reward, done, _ = env.step(a)
        U[ob] += alpha * (reward + gamma * U[new_ob] - U[ob])
        if done:
            ob = env.reset()
        else:
            ob = new_ob
    
    return U


if __name__ == '__main__':
    environment = "FrozenLake-v1"
    env = gym.make(environment)
    env.seed(0)
    gamma = 0.9
    tolerance = 1e-3
    alpha = 0.2

    value_function, policy = policy_iteration(
        env.env.P, env.env.nS, env.env.nA, gamma, tolerance)

    est_val_func = td_policy_eval(policy, env.env, gamma, alpha)
    # print(value_function)
    # heatmap_value_func(value_function.reshape(8 , -1))
    print(est_val_func)
    heatmap_value_func(est_val_func.reshape(4 , -1))
