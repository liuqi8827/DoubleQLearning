from collections import defaultdict
from tqdm import tqdm as _tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import random

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class SingleDQN():

    def __init__(self):
        pass

    def get_epsilon(self, it):
        if it >= 1000:
            return 0.05
        else:
            return 1 - (0.00095 * it)

    class QNetwork(nn.Module):
        def __init__(self, num_hidden=128):
            nn.Module.__init__(self)
            self.l1 = nn.Linear(4, num_hidden)
            self.l2 = nn.Linear(num_hidden, 2)

        def forward(self, x):
            out = self.l1(x)
            out = F.relu(out)
            out = self.l2(out)
            return out

    class ReplayMemory:
        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []

        def push(self, transition):
            if len(self.memory) < self.capacity:
                self.memory.append(transition)

        def sample(self, batch_size):
            return random.sample(self.memory, k=batch_size)

        def __len__(self):
            return len(self.memory)


    class EpsilonGreedyPolicy():
        """
        A simple epsilon greedy policy.
        """
        def __init__(self, Q, epsilon):
            self.Q = Q
            self.epsilon = epsilon
        
        def sample_action(self, obs):
            """
            This method takes a state as input and returns an action sampled from this policy.  

            Args:
                obs: current state

            Returns:
                An action (int).
            """
            if np.random.uniform() < self.epsilon:
                return np.random.randint(2)

            with torch.no_grad():
                out = self.Q(torch.tensor(obs, dtype=torch.float))
                return out.argmax().item()

        def set_epsilon(self, epsilon):
            self.epsilon = epsilon
        
    def compute_q_vals(self, Q, states, actions):
        """
        This method returns Q values for given state action pairs.
        
        Args:
            Q: Q-net
            states: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: Shape: batch_size x 1

        Returns:
            A torch tensor filled with Q values. Shape: batch_size x 1.
        """
        values = Q(states.float())
        return torch.gather(values, 1, actions)
        
    def compute_targets(self, Q, rewards, next_states, dones, discount_factor):
        """
        This method returns targets (values towards which Q-values should move).
        
        Args:
            Q: Q-net
            rewards: a tensor of actions. Shape: Shape: batch_size x 1
            next_states: a tensor of states. Shape: batch_size x obs_dim
            dones: a tensor of boolean done flags (indicates if next_state is terminal) Shape: batch_size x 1
            discount_factor: discount
        Returns:
            A torch tensor filled with target values. Shape: batch_size x 1.
        """

        # process next_states
        max_values, _ = Q(next_states.float()).max(dim=1)
        max_values = max_values.view(-1, 1)
        
        # Set the max values of all states that will be done to 1
        max_values = torch.where(dones == 1, torch.zeros_like(max_values), max_values)
        target = rewards + discount_factor * max_values

        return target
        
    def train(self, Q, memory, optimizer, batch_size, discount_factor):
        # DO NOT MODIFY THIS FUNCTION
        
        # don't learn without some decent experience
        if len(memory) < batch_size:
            return None

        # random transition batch is taken from experience replay memory
        transitions = memory.sample(batch_size)
        
        # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
        state, action, reward, next_state, done = zip(*transitions)
        
        # convert to PyTorch and define types
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.int64)[:, None]  # Need 64 bit to use them as index
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)[:, None]
        done = torch.tensor(done, dtype=torch.uint8)[:, None]  # Boolean
        
        # compute the q value
        q_val = self.compute_q_vals(Q, state, action)
        with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
            target = self.compute_targets(Q, reward, next_state, done, discount_factor)
        
        # loss is measured from error between current and newly expected Q values
        loss = F.smooth_l1_loss(q_val, target)

        # backpropagation of loss to Neural Network (PyTorch magic)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

    def run_episodes(self, Q, policy, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
        optimizer = optim.Adam(Q.parameters(), learn_rate)
        
        global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
        episode_durations = []  #
        for i in range(num_episodes):
            state = env.reset()
            steps = 0
            while True:
                action = policy.sample_action(state)
                next_state, reward, done, _ = env.step(action)
                
                memory.push((state,action,reward,next_state,done))
                
                steps += 1
                
                global_steps += 1
                policy.set_epsilon(self.get_epsilon(global_steps))
                
                self.train(Q, memory, optimizer, batch_size, discount_factor) 
                
                if done:
                    if i % 10 == 0:
                        print("{2} Episode {0} finished after {1} steps"
                            .format(i, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                    episode_durations.append(steps)
                    #plot_durations()
                    break
                
                state = next_state
            
        return episode_durations