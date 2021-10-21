from collections import defaultdict
from tqdm import tqdm as _tqdm
import numpy as np


def tqdm(*args, **kwargs):
    # Safety, do not overflow buffer
    return _tqdm(*args, **kwargs, mininterval=1)


class QLearning(object):

    """
    Base Class for Single and double Q-learning. Note how Q-values are provided by means of the policy.

    Methods:
        update_Q: the function for updating the Q-values
        final_Q: returns a function of all the Q_values at the end. For example: 
            for double Q-learning this could return the sum of Q_a and Q_b.
        train: the training loop for Q-learning.

    """

    def __init__(self):
        pass

    def update_Q(self, policy, reward, state, next_state, action):
        pass

    def final_Q(self, policy):
        pass


    def train(self, env, policy, num_episodes, discount_factor=1.0, alpha=0.5, show_episodes=True):
        """
        Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
        while following an epsilon-greedy policy

        Args:
            env: OpenAI environment.
            policy: An epsilon-greedy policy which allows us to sample episodes with its sample_episode method.
                    Note that the policy (and its tabular Q-value function) should already be initialized.
            num_episodes: Number of episodes to run for.
            discount_factor: Gamma discount factor.
            alpha: TD learning rate.
            show_episodes: whether to show a progress_bar for sampling episodes

        Returns:
            A tuple (Q, stats).
            Q is a numpy array Q[s,a] -> state-action value.
            stats is a list of tuples giving the episode lengths and returns.
        """
        # Set alpha and discount_factor for later use by update_Q
        self.alpha = alpha
        self.discount_factor = discount_factor

        # Keep track of useful statistics
        stats = []

        # Optional progress bar
        episode_range = tqdm(range(num_episodes)) if show_episodes else range(num_episodes)

        Q_values = []
        for i_episode in episode_range:
            i, R = policy.sample_episode(env, self.update_Q)
            stats.append((i, R))

            Q_values.append([policy.Q_a[0][0], policy.Q_a[0][1]])
        episode_lengths, episode_returns = zip(*stats)

        Q = self.final_Q(policy)

        return Q, (episode_lengths, episode_returns), Q_values


class SingleQLearning(QLearning):



    def __init__(self):
        super().__init__()

    def update_Q(self, policy, reward, state, next_state, action):
        """
        Applies the update rule for single Q-learning to the policy's Q-values
        """
        Q = policy.Q_a

        max_action = np.argmax(Q[next_state, :])
        Q[state, action] = Q[state, action] + self.alpha * \
            (reward + self.discount_factor * Q[next_state, max_action] - Q[state, action])


    def final_Q(self, policy):
        return policy.Q_a


class DoubleQLearning(QLearning):
    def __init__(self):
        super().__init__()

    def update_Q(self, policy, reward, state, next_state, action):
        """
        Applies the update rule for double Q-learning to the policy's Q-values.
        """
        Q_a = policy.Q_a
        Q_b = policy.Q_b
        if np.random.randint(2):
            self.update_double_Q(Q_a, Q_b, reward, state, next_state, action)
        else:
            self.update_double_Q(Q_b, Q_a, reward, state, next_state, action)

    def update_double_Q(self, Q_1, Q_2, reward, state, next_state, action):

        max_action = np.argmax(Q_1[next_state, :])
        Q_1[state, action] = Q_1[state, action] + self.alpha * \
            (reward + self.discount_factor * Q_2[next_state, max_action] - Q_1[state, action])

        return


    def final_Q(self, policy):
        return policy.Q_a + policy.Q_b
