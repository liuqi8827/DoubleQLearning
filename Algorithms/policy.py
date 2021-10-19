import numpy as np


class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy, using a table of Q-values.
    args:
        Q_a : numpy array of size(num_states, n_actions)
        Q_b : Either a numpy array of size(num_states, n_actions) (for double Q-learning), 
                or None (for Single Q_learning).
        epsilon: the epsilon (float [0-1])
    """

    def __init__(self, epsilon, Q_a, Q_b=None):
        self.Q_a = Q_a
        self.Q_b = Q_b
        self.epsilon = epsilon

    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: current state

        Returns:
            An action (int).
        """

        action_values = self.Q_a[obs]
        if self.Q_b is not None:
            action_values += self.Q_b[obs]

        if (np.random.uniform(0, 1) <= self.epsilon):
            action = np.random.randint(0, len(action_values))
        else:
            action = np.argmax(action_values)

        return action


    def sample_episode(self, env, update_func=None):
        """
        Samples an episode for a given env.
        If update_func is provided, also updates the Q-values

        Args:
            env: The OpenAI environment to sample from
            update_func: The update function for the Q-values

        Returns:
            i: the episode length
            R: the episode return
        """
        
        assert (env.nS, env.nA) == self.Q_a.shape, "Environment and Q-table dimensions do not align."
        i = 0
        R = 0
        state = env.reset()
        done = False

        while not done:
            action = self.sample_action(state)
            transition = env.step(action)
            next_state, reward, done, _ = transition

            if update_func:
                update_func(self, reward, state, next_state, action)

            state = next_state

            R += reward
            i += 1

        return (i, R)
