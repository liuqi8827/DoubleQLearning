import numpy as np


class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy, using Q-values.
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
        assert (env.nS, env.nA) == self.Q_a.shape, "Matrix dimensions do not align."
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
