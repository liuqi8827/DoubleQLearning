from collections import defaultdict
from tqdm import tqdm as _tqdm
import numpy as np

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class QLearning(object):

    def __init__(self):
        pass

    def update_Q(self, Q_1, Q_2, reward, state, next_state, action):
        pass

    def final_Q(self, Q_a, Q_b):
        pass

    def new_val(self):
        return np.zeros(val)

    def get_num_actions(self, env):
        try:
            num_actions = env.action_space.n
        except AttributeError:
            num_actions = env.nA

        return num_actions


    def sample_episode(self, env, policy, Q_a, Q_b):
        i = 0
        R = 0

        state = env.reset()
        if isinstance(env.nA, list):
            val = env.nA[state]
        else:
            val = env.nA

        Q_a.setdefault(state,  self.new_val(val))
        Q_b.setdefault(state, self.new_val(val))

        while not done:
            action = policy.sample_action(state)
            transition = env.step(action)
            next_state, reward, done, _ = transition

            self.update_Q(Q_a, Q_b, reward, state, next_state, action)

            state = next_state

            R += reward
            i += 1

        return (i, R)

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
        self.num_actions = self.get_num_actions(env)


        Q_a = policy.Q_a
        Q_b = policy.Q_b


        episode_range = tqdm(range(num_episodes)) if show_episodes else range(num_episodes)

        for i_episode in episode_range:
            i, R = self.sample_episode(env, policy, Q_a, Q_b)
            stats.append((i, R))

        episode_lengths, episode_returns = zip(*stats)

        Q = self.final_Q(Q_a, Q_b)

        return Q, (episode_lengths, episode_returns)




class SingleQLearning(QLearning):

    def __init__(self):
        super().__init__()


    def update_Q(self, Q_a, Q_b, reward, state, next_state, action):
        Q = Q_a

        if isinstance(env.nA, list):
            val = env.nA[state]
        else:
            val = env.nA

        max_action = np.argmax(Q.setdefault(next_state, self.new_val(val)))
        Q[state][action] = Q[state][action] + self.alpha * (reward + self.discount_factor * Q[next_state][max_action] - Q[state][action])


    def final_Q(Q):
        return Q



class DoubleQLearning(QLearning):
    def __init__(self):
        super().__init__()


    def update_Q(self, Q_a, Q_b, reward, state, next_state, action):
        if np.random.randint(2):
            self.update_double_Q(Q_a, Q_b, reward, state, next_state, action)
        else:
            self.update_double_Q(Q_b, Q_a, reward, state, next_state, action)


    def update_double_Q(self, Q_1, Q_2, reward, state, next_state, action):
        if isinstance(env.nA, list):
            val = env.nA[state]
        else:
            val = env.nA
        max_action = np.argmax(Q_1.setdefault(next_state, self.new_val(val)))
        Q_1[state][action] = Q_1[state][action] + self.alpha * \
            (reward + self.discount_factor * Q_2.setdefault(next_state, self.new_val(val))[max_action] - Q_1[state][action])
        return

    def final_Q(Q_a, Q_b):
        Q = {}
        for key in Q_a.keys():
            Q[key] = Q_a[key] + Q_b[key]
        return Q
