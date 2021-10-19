from collections import defaultdict
from tqdm import tqdm as _tqdm
import numpy as np

from Algorithms.policy import EpsilonGreedyPolicy

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class QLearning(object):

    def __init__(self):
        pass

    def update_Q(self, reward, state, next_state, action):
        pass

    def final_Q(self, policy):
        pass


    def train(self, env, policy, num_episodes, discount_factor=1.0, alpha=0.5, show_episodes=True):
        """
        Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
        while following an epsilon-greedy policy

        Args:
            env: OpenAI environment.
            policy: A behavior policy which allows us to sample actions with its sample_action method.
            Q: Q value function
            num_episodes: Number of episodes to run for.
            discount_factor: Gamma discount factor.
            alpha: TD learning rate.

        Returns:
            A tuple (Q, stats).
            Q is a numpy array Q[s,a] -> state-action value.
            stats is a list of tuples giving the episode lengths and returns.
        """

        # Keeps track of useful statistics
        stats = []
        self.env = env
        self.alpha = alpha
        self.discount_factor = discount_factor
        self.nA = env.nA

        episode_range = tqdm(range(num_episodes)) if show_episodes else range(num_episodes)

        for i_episode in episode_range:
            i, R = policy.sample_episode(env, self.update_Q)
            stats.append((i, R))

        episode_lengths, episode_returns = zip(*stats)

        Q = self.final_Q(policy)

        return Q, (episode_lengths, episode_returns)




class SingleQLearning(QLearning):

    def __init__(self):
        super().__init__()


    def update_Q(self, policy, reward, state, next_state, action):
        Q = policy.Q_a
        max_action = np.argmax(Q[next_state, :])
        Q[state, action] = Q[state, action] + self.alpha * (reward + self.discount_factor * Q[next_state, max_action] - Q[state, action])


    def final_Q(self, policy):
        return policy.Q_a



class DoubleQLearning(QLearning):
    def __init__(self):
        super().__init__()


    def update_Q(self, policy, reward, state, next_state, action):
        Q_a = policy.Q_a
        Q_b = policy.Q_b
        if np.random.randint(2):
            self.update_double_Q(Q_a, Q_b, reward, state, next_state, action)
        else:
            self.update_double_Q(Q_b, Q_a, reward, state, next_state, action)


    def update_double_Q(self, Q_1, Q_2, reward, state, next_state, action):
        max_action = np.argmax(Q_1[next_state, :])
        Q_1[state, action] = Q_1[state, action] + self.alpha * \
            (reward + self.discount_factor * Q_2[next_state, max_action] - Q_1[state][action])
        return


    def final_Q(self, policy):
        return policy.Q_a + policy.Q_b
