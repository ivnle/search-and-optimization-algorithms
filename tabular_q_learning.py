# %%
import gym
import numpy as np
from policy_iteration import policy_iteration, heatmap_value_func


def pick_action(s, eps, env, Q):
    if np.random.random() < eps:
        return env.action_space.sample()        
    else:
        return np.argmax(Q[s])


def deep_q_learning(env, gamma, alpha, iterations, eps=0.1):
    """
    Goal: Estimate the optimal policy without knowing the transition probabilities.
    """
    
    Q = np.zeros((env.nS, env.nA))
    training_rewards = []
    episode_rewards = 0

    ob = env.reset()
    for i in range(iterations):    
        a = pick_action(ob, eps, env, Q)
        new_ob, reward, done, _ = env.step(a)
        Q[ob, a] += alpha * (reward + gamma * np.max(Q[new_ob]) - Q[ob, a])
        episode_rewards += reward
        if done:
            ob = env.reset()
            training_rewards.append(episode_rewards)
            episode_rewards = 0
        else:
            ob = new_ob
    
    return Q, training_rewards


if __name__ == '__main__':
    environment = "FrozenLake-v1"
    # environment = "FrozenLake8x8-v0"
    # environment = "Taxi-v2"
    env = gym.make(environment)
    env.reset
    gamma = 0.9
    alpha = 0.7
    eps = 0.4
    iterations = 1000000
    env.render()
    Q, training_rewards = deep_q_learning(env.env, gamma, alpha, iterations, eps)
    policy = np.argmax(Q, axis=1)
    value_function = np.max(Q, axis=1)
    # print(value_function)
    heatmap_value_func(value_function.reshape(int(env.env.nS**(1/2)) , -1))
    # print(est_val_func)
    # heatmap_value_func(est_val_func.reshape(8 , -1))
