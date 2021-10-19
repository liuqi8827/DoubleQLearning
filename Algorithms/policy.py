import numpy as np

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, epsilon, Q_a : dict, Q_b: dict = None):
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
        if self.Q_b:
            action_values = self.Q_a[obs] + self.Q_b[obs]
        else:
            action_values = self.Q_a[obs]

        if (np.random.uniform(0, 1) <= self.epsilon):
            action = np.random.randint(0, len(action_values))
        else:
            action = np.argmax(action_values)
                        
        return action