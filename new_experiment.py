# %%

import numpy as np
import matplotlib.pyplot as plt

# %%
# Initial Values
terminal_states = [2, 3]
gamma = 1


# %%
# Create Q


def is_terminal(state):
    return state in terminal_states


def initial_Q(possible_actions):
    return [[0 for _ in possible_actions[i]] for i in range(4)]


def make_transition(state, action, std=1):
    if state == 0:
        if action == 0:
            return 0, 1
        if action == 1:
            return 0, 2

    if state == 1:
        return np.random.normal(-0.5, std), 3

    # If a move is made from any other state, nothing happens
    return 0, state


# %%
# Update Q-values

class SingleQ():

    def __init__(self, Q):
        self.Q = Q

    def sample_action(self, state, epsilon=0.1):

        if np.random.random() < epsilon:
            return np.random.choice(range(len(self.Q[state])))
        else:
            return np.argmax(self.Q[state])

    def getMaxQ(self, state):
        best_move = np.argmax(self.Q[state])

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
            return np.argmax(Q_s)

    def updateQ(self, state, action, reward, next_state, alpha=0.1):

        if np.random.randint(0, 2):
            best_move = np.argmax(self.Q_a[next_state])

            self.Q_a[state][action] = self.Q_a[state][action] + alpha * \
                (reward + gamma * self.Q_b[next_state][best_move]
                 - self.Q_a[state][action])

        else:
            best_move = np.argmax(self.Q_b[next_state])

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

    left_count = 0
    left_counts = []

    # number of states visited
    ns = [0 for _ in range(len(possible_actions))]
    # number of state-action visited
    nsa = [[0 for _ in possible_actions[state]]
           for state in range(len(possible_actions))]

    for i in range(1, episodes+1):
        state = 0

        while True:
            # increment the state visits
            ns[state] += 1

            # determine the next action
            action = learner.sample_action(state, 1/np.sqrt(ns[state]))

            # increment left actions
            if state == 0 and action == 0:
                left_count += 1
            # increment state-action visits
            nsa[state][action] += 1

            # make transition
            reward, next_state = make_transition(state, action, std)

            # determine alpha
            alpha = 1 / np.power(nsa[state][action], .8)

            # update Q-value
            Q = learner.updateQ(state, action, reward, next_state, alpha)

            # break if a terminal state is reached
            if is_terminal(next_state):
                break

            state = next_state

        left_counts.append(left_count)

    return left_counts


# %%

def experiment(num_experiments=100, episodes=300, num_random_actions=5, std=1, doubleQ=False):
    final_count = [0 for i in range(episodes)]
    final_res = [0 for i in range(episodes)]

    possible_actions = [[0, 1], list(range(num_random_actions)), [0], [0]]

    for i in range(num_experiments):
        current_count = single_experiment(
            possible_actions, episodes, std, doubleQ)

        for e in range(episodes):
            final_count[e] = (final_count[e] * i + current_count[e]) / (i+1)
            final_res[e] = final_count[e] / (e+1) * 100

    return final_res


num_random_actions = 5
std = 2
single_res = experiment(num_random_actions=num_random_actions, std=std)
double_res = experiment(
    num_random_actions=num_random_actions, std=std, doubleQ=True)

plt.plot(range(len(single_res)), single_res, label="single")
plt.plot(range(len(double_res)), double_res, label="double")
plt.legend()
plt.xlabel("episodes")
plt.ylabel("percentage left")

# %%
