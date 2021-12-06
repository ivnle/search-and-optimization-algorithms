# %%
import gym
import numpy as np
import torch
from torch import nn
from policy_iteration import policy_iteration, heatmap_value_func
import random
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import math

class LinearNN(nn.Module):
    """
    3 layer densely connected network.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, action_dim)
        )

    def forward(self, x):
        return self.linear(x)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        # Get all the samples if the requested batch_size is higher than the number of sample currently in the memory
        # batch_size = min(batch_size, len(self))
        # Randomly select "batch_size" samples and return the selection
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def choose_action(net, state, epsilon, device):
    
    with torch.no_grad():
        net.eval()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        q_values = net(state)

    best_action = int(q_values.argmax())
    action_dim = q_values.shape[-1]

    # Select a non optimal action with probability epsilon, otherwise choose the best action
    if random.random() < epsilon:
        action = random.randrange(action_dim)
    else:
        action = best_action

    return action, q_values.cpu().numpy()


def update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size):
    pass

def update_epsilon(epsilon, steps):
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps / EPS_DECAY)
    return eps_threshold


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # environment = 'FrozenLake-v0'
    # environment = 'Acrobot-v1'
    #environment = "FrozenLake8x8-v1"
    #environment = "Taxi-v3"
    environment = "CartPole-v1"

    env = gym.make(environment)
    state_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.n

    # Set random seeds
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # Set parameters
    gamma = 0.99
    epsilon = 0.1
    replay_memory_capacity = 10000
    lr = 1e-3
    target_net_update_steps = 10
    batch_size = 256
    episodes = 200
    min_samples_for_training = 1000

    # Initialize replay memory
    replay_mem = ReplayMemory(replay_memory_capacity)

    # Initialize policy and target networks. The target network shares the same initial weights as the policy network.
    policy_net = LinearNN(state_space_dim, action_space_dim).to(device)
    target_net = LinearNN(state_space_dim, action_space_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # Only optimize the parameters of the policy network
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    # Use Huber loss: https://en.wikipedia.org/wiki/Huber_loss
    loss_fn = nn.SmoothL1Loss()

    reward_hist = []
    steps = 0

    for episode_num in tqdm(range(episodes)):
        state = env.reset()
        score = 0
        done = False

        while not done:
            action, q_values = choose_action(policy_net, state, epsilon, device)

            steps += 1
            epsilon = update_epsilon(epsilon, steps)
            
            next_state, reward, done, info = env.step(action)

            replay_mem.push(state, action, next_state, reward)
            score += reward

            if len(replay_mem) > min_samples_for_training:
                update_step(policy_net, target_net, replay_mem,
                            gamma, optimizer, loss_fn, batch_size)

            state = next_state

        if episode_num % target_net_update_steps == 0:
            target_net.load_state_dict(policy_net.state_dict())

        reward_hist.append(score)

    env.close()

    plt.plot(reward_hist)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    #save plot
    plt.savefig('figures/deep_q_learning.png')
