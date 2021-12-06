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


def monte_carlo_policy_eval(policy, env, gamma):
    """
    Goal: Estimate value of a policy without knowing the transition probabilities.
    """
    # Dict form state to sequence of reward values
    G = defaultdict(list)

    # Value function we are trying to approximate
    U = np.zeros(env.nS)

    iterations = 1000
    max_steps = 100

    for i in range(iterations):
        ob = env.reset()
        episode = [(ob, 0)]
        for t in range(max_steps):
            a = policy[ob]
            ob, reward, done, _ = env.step(a)
            episode.append((ob, reward))
            # episode_reward += rew
            if done:
                break
        for k, (s, r) in enumerate(episode):
            G[s].append(reward_to_go(s, k, episode, gamma))
            U[s] = np.mean(G[s])

    return U


if __name__ == '__main__':
    environment = "FrozenLake-v1"
    env = gym.make(environment)
    
    gamma = 0.9
    tolerance = 1e-3

    value_function, policy = policy_iteration(
        env.env.P, env.env.nS, env.env.nA, gamma, tolerance)

    est_val_func = monte_carlo_policy_eval(policy, env.env, gamma)
    # print(value_function)
    # heatmap_value_func(value_function.reshape(8 , -1))
    print(est_val_func)
    heatmap_value_func(est_val_func.reshape(4 , -1))
