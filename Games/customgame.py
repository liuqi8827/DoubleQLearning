# %%

# from gridworld import GridworldEnv
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box

# %%

# 0 is left, 1 is right


class FirstGame(Env):
    """Simple game that should show maximization bias.

    The player starts in state 0.
    The player has two choices, left or right.
    If left is chosen for state 0 the player is rewarded 0.6 and a terminal state is reached.
    If right is chosen the player is rewarded 0 and moves to state 2.

    From state 2 the player both left and right will result in move 3. 
    The reward is sampled from a uniform distribution between 0 and 1.
    """

    def __init__(self, num_actions):
        self.nA = [2, 1, num_actions, 1]
        self.nS = 4

        self.transitions = {
            0: {0: (1, (0.6, 0), True), 1: (2, (0, 0), False)},
            2: {k: (3, (0.5, 1), True) for k in range(num_actions)}}

        self.s = 0
        self.is_done = False

    def get_transition(self, action):

        next_state, bounds, done = self.transitions[self.s][action]

        return next_state, np.random.normal(bounds[0], bounds[1]), done

    def step(self, action):

        if self.is_done:
            return self.s, 0, True, 1.0

        next_state, reward, done = self.get_transition(action)

        self.s = next_state
        self.is_done = done

        return next_state, reward, done, 1.0

    def reset(self):
        self.s = 0
        self.is_done = False

        return self.s


# %%


game = FirstGame(100)

# %%

game.transitions

# %%

game.reset()
