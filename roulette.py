# %%

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
# Initial Values
terminal_states = [1]

# %%
# Create Q


def is_terminal(state):
    return state in terminal_states


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


def initial_Q(possible_actions):
    return [[0 for _ in possible_actions[i]] for i in range(len(possible_actions))]


def make_transition(state, action):
    max_move = 10
    if state == 0 and action == max_move:
        return 0, 1

    if state == 0:
        # return np.random.normal(-0.027, 1), 0

        throw = np.random.randint(0, max_move)

        if throw == action:
            return max_move-2, 0

        return -1, 0

    # If a move is made from any other state, nothing happens
    return 0, state


# loops = 10000
# for a in range(10):
#     reward = 0
#     for i in range(loops):
#         r, _ = make_transition(0, a)

#         reward += r

#     print(f"action: {a} => {reward/loops}")

# %%

# ints = {k: 0 for k in range(36)}
# for i in range(10000):
#     ints[np.random.randint(0, 36)] += 1

# print(ints)

# %%
# Update Q-values

gamma = 0.9


class SingleQ():

    def __init__(self, Q):
        self.Q = Q

    def sample_action(self, state, epsilon=0.1):

        if np.random.random() < epsilon:
            return np.random.choice(range(len(self.Q[state])))
        else:
            return my_arg_max(self.Q[state])

    def getMaxQ(self, state):
        best_move = my_arg_max(self.Q[state])

        return best_move, self.Q[state][best_move]

    def updateQ(self, state, action, reward, next_state, alpha=0.1):

        _, next_value = self.getMaxQ(next_state)

        self.Q[state][action] = self.Q[state][action] + alpha * \
            (reward + gamma * next_value - self.Q[state][action])

# %%


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


def single_experiment(possible_actions, episodes, std=1, doubleQ=False):
    if doubleQ:
        learner = DoubleQ(initial_Q(possible_actions),
                          initial_Q(possible_actions))

    else:
        learner = SingleQ(initial_Q(possible_actions))

    payout = 0
    payouts = []

    episode_lengths = []

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

    return payouts, episode_lengths


# %%

def experiment(num_experiments=500, episodes=300, std=1, doubleQ=False):
    final_count = [0 for i in range(episodes)]
    final_res = [0 for i in range(episodes)]
    final_t = [0 for i in range(episodes)]

    possible_actions = [list(range(11)), [0]]

    first_count = []

    for i in range(num_experiments):
        current_count, current_t = single_experiment(
            possible_actions, episodes, std, doubleQ)

        first_count.append(current_t[0])
        # break

        for e in range(episodes):
            final_count[e] = (final_count[e] * i + current_count[e]) / (i+1)
            final_res[e] = final_count[e] / (e+1) * 100

            final_t[e] = (final_t[e] * i + current_t[e]) / (i+1)

    return final_res, final_t


num_experiments = 10
episodes = 100
single_res, single_t = experiment(
    episodes=episodes, num_experiments=num_experiments)
double_res, double_t = experiment(
    episodes=episodes, num_experiments=num_experiments, doubleQ=True)

plt.plot(range(len(single_res)), single_t, label="single")
plt.plot(range(len(double_res)), double_t, label="double")
plt.legend()
plt.xlabel("episodes")
plt.ylabel("percentage left")

# %%

# variables to change: std, number of actions

# Experiment 1: number of actions

# single

for n in [1, 2, 3, 4, 5, 10, 20, 50, 100]:

    res = experiment(num_random_actions=n)

    plt.plot(range(len(res)), res, label=n)

plt.legend()
plt.xlabel("episodes")
plt.ylabel("percentage left")

plt.show()
# %%
# double

for n in [1, 2, 3, 4, 5, 10, 20, 50, 100]:

    res = experiment(num_random_actions=n, doubleQ=True)

    plt.plot(range(len(res)), res, label=n)

plt.legend()
plt.xlabel("episodes")
plt.ylabel("percentage left")

plt.show()
# %%

# variables to change: std, number of actions

# Experiment 2: std

# single

for s in [0, 0.5, 1, 2, 3, 10]:

    res = experiment(std=s)

    plt.plot(range(len(res)), res, label=s)

plt.legend()
plt.xlabel("episodes")
plt.ylabel("percentage left")

plt.show()
# %%

# variables to change: std, number of actions

# Experiment 2: std

# double

for s in [0, 0.5, 1, 2, 3]:

    res = experiment(std=s, doubleQ=True)

    plt.plot(range(len(res)), res, label=s)

plt.legend()
plt.xlabel("episodes")
plt.ylabel("percentage left")

plt.show()
# %%

# Image David


res = experiment(std=0)
plt.plot(range(len(res)), res, label="deterministic")


res = experiment(std=1)
plt.plot(range(len(res)), res, label="stochastic")


plt.legend()
plt.xlabel("episodes")
plt.ylabel("percentage left")

plt.show()

# %%

(1/36 * 34) - (35/36 * 1)
# %%
