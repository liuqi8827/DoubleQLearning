# %%

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# %%
# Initial Values
terminal_states = [1]
gamma = 0.95

# Ensure save folder for figures exist
os.makedirs('Results', exist_ok=True)
# %%

def is_terminal(state):
    return state in terminal_states


# The builtin np.argmax always chooses the left-most value if there are multiple maxima, so we define a replacement
def my_arg_max(actions):
    arg_max = [0]
    value = actions[0]

    for i in range(1, len(actions)):
        if (actions[i]) > value:
            value = actions[i]
            arg_max = [i]

        if (actions[i] == value):
            arg_max.append(i)

    return np.random.choice(arg_max)


# Create Q
def initial_Q(possible_actions):
    return [[0 for _ in possible_actions[i]] for i in range(len(possible_actions))]


# Define the transitions of the environment
def make_transition(state, action):
    max_move = 36

    if state == 0 and action == max_move:
        return 0, 1

    if state == 0:
        # return np.random.normal(-0.027, 1), 0

        throw = np.random.randint(0, max_move)

        if throw == action:
            return max_move, 0

        return -1, 0

    # If a move is made from any other state, nothing happens
    return 0, state

# %%
# Single Q-learning


class SingleQ():

    def __init__(self, Q):
        self.Q = Q

    def sample_action(self, state, epsilon=0.1):

        if np.random.random() < epsilon:
            return np.random.choice(range(len(self.Q[state])))
        else:
            return my_arg_max(self.Q[state])

    def getMaxQ(self, state):
        best_move = np.argmax(np.array(self.Q[state]))

        return best_move, self.Q[state][best_move]

    def updateQ(self, state, action, reward, next_state, alpha=0.1):

        _, next_value = self.getMaxQ(next_state)

        self.Q[state][action] = self.Q[state][action] + alpha * \
            (reward + gamma * next_value - self.Q[state][action])

# %%

# Double Q-learning
class DoubleQ():

    def __init__(self, Q_a, Q_b):
        self.Q_a = Q_a
        self.Q_b = Q_b

    def sample_action(self, state, epsilon=0.1):

        if np.random.random() < epsilon:
            return np.random.choice(range(len(self.Q_a[state])))
        else:
            Q_s = np.array(self.Q_a[state]) + np.array(self.Q_b[state])
            return my_arg_max(Q_s)

    def updateQ(self, state, action, reward, next_state, alpha=0.1):

        if np.random.randint(0, 2):
            best_move = my_arg_max(self.Q_a[next_state])

            self.Q_a[state][action] = self.Q_a[state][action] + alpha * \
                (reward + gamma * self.Q_b[next_state][best_move]
                 - self.Q_a[state][action])

        else:
            best_move = my_arg_max(self.Q_b[next_state])

            self.Q_b[state][action] = self.Q_b[state][action] + alpha * \
                (reward + gamma * self.Q_a[next_state][best_move]
                 - self.Q_b[state][action])


# %%

# %%



def single_experiment(possible_actions, episodes, doubleQ=False):
    """
    This runs an experiment for a single seed.
    We track number of states and state-action pairs visited for the adaptive epsilon and alpha.
    Args:
        possible_actions: list of possible actions (list(int))
        episodes: number of episodes (int)
        std: standard deviation (float)
        doubleQ: whether to use double Q-learning instead of single (Bool)

    Returns:
        payouts: list of payouts, i.e. return per episode (int)
        episode_lenghts: list of episode lengths (int)
        average_q_vals: list of the average q-value per episode (list(float))
    """
    if doubleQ:
        learner = DoubleQ(initial_Q(possible_actions),
                          initial_Q(possible_actions))

    else:
        learner = SingleQ(initial_Q(possible_actions))

    payouts = []
    payout = 0

    episode_lengths = []
    average_q_vals = []

    # number of states visited
    ns = [0 for _ in range(len(possible_actions))]
    # number of state-action visited
    nsa = [[0 for _ in possible_actions[state]]
           for state in range(len(possible_actions))]

    for i in tqdm(range(1, episodes+1)):
        state = 0
        moves_count = 0

        while True:
            moves_count += 1

            # increment the state visits
            ns[state] += 1

            # determine the next action
            action = learner.sample_action(state, 1/np.sqrt(ns[state]))

            # increment state-action visits
            nsa[state][action] += 1

            # make transition
            reward, next_state = make_transition(state, action)
            # print(f"action: {action}, reward: {reward}")

            payout += reward
            # determine alpha
            alpha = 1 / np.power(nsa[state][action], .8)

            # update Q-value
            learner.updateQ(state, action, reward, next_state, alpha)

            # break if a terminal state is reached
            if is_terminal(next_state):
                episode_lengths.append(moves_count)
                break

            state = next_state

        # print(f"Q at {i}: {learner.Q}")
        payouts.append(payout)
        if doubleQ:
            average_q_vals.append(sum(learner.Q_a[0]) / len(learner.Q_a[0]))
        else:
            average_q_vals.append(sum(learner.Q[0]) / len(learner.Q[0]))

    return payouts, episode_lengths, average_q_vals


# %%


def experiment(num_experiments=500, episodes=300, doubleQ=False):
    """ 
    Runs an experiment N times, with a different seed each time.
    Args:
        num_experiments: number of experiments (int)
        episodes: number of episodes per experiment (int)
        std: standard deviation (float)
        doubleQ: whether to use double Q-learning instead of single (Bool)

    Returns:
        final_res: numpy array of size (num_experimentes x episodes), 
            containing the average q-values per episode for each experiment
    """
    final_count = [0 for i in range(episodes)]
    final_res = [0 for i in range(episodes)]
    final_t = [0 for i in range(episodes)]

    final_res = []

    possible_actions = [list(range(37)), [0]]

    first_count = []

    for i in range(num_experiments):
        np.random.seed(i)
        print(f"Starting exerperiment: {i}")
        current_count, current_t, average_q = single_experiment(
            possible_actions, episodes, doubleQ)

        first_count.append(current_t[0])
        final_res.append(np.array(average_q))

        # for e in range(episodes):
        #     final_count[e] = (final_count[e] * i + current_count[e]) / (i+1)
        #     final_res[e] = final_count[e] / (e+1) * 100

        #     final_t[e] = (final_t[e] * i + current_t[e]) / (i+1)

    # total_average_q = np.mean(np.array(total_average_q), axis=0)
    return np.array(final_res)


num_experiments = 20
episodes = 100
single_res = experiment(
    episodes=episodes, num_experiments=num_experiments)
double_res = experiment(
    episodes=episodes, num_experiments=num_experiments, doubleQ=True)

# %%
# Plot average and standard deviation of the expected result for each episode for single and double q-learning

m = single_res.mean(axis=0)
s = single_res.std(axis=0)/2

plt.plot(range(len(single_res[0])), m, label="single")
plt.fill_between(range(len(single_res[0])), m-s, m+s, alpha=0.25)

m = double_res.mean(axis=0)
s = double_res.std(axis=0)
plt.plot(range(len(double_res[0])), m, label="double")
plt.fill_between(range(len(single_res[0])), m-s, m+s, alpha=0.25)

plt.legend(bbox_to_anchor=(1.3, 1.0))

plt.xlabel("episodes")
plt.ylabel("expected profit")

plt.tight_layout()
plt.savefig("Results/roulette.png")


# %%

# %%

# Plot expected results for each episode separately for each seed.
plt.plot(range(len(single_res)), single_res, label="single")
plt.plot(range(len(double_res)), double_res, label="double")

plt.legend(bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.xlabel("episodes")
plt.ylabel("expected profit")

# %%
