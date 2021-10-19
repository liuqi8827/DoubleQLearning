# %%

import numpy as np
import matplotlib.pyplot as plt

# %%
# Initial Values

num_experiments = 100
iterations_per_experiments = 300
num_actions = 5
possible_actions = [[0, 1], list(range(num_actions)), [0], [0]]

terminals = [2, 3]

gamma = 1
epsilon = 0.1

# %%
# Create Q


def is_terminal(state):
    return state in terminals


def initial_Q(possible_actions):
    return [[0 for _ in possible_actions[i]] for i in range(4)]


Q = initial_Q(possible_actions)

# %%
# action sampler


def sample_action(Q, state, epsilon=0.1):

    if np.random.random() >= epsilon:
        print()
        return np.random.choice(range(len(Q[state])))
    else:
        return np.argmax(Q[state])


# %%
# Transition functions


def make_transition(state, action):
    if state == 0:
        if action == 0:
            return 0, 1
        if action == 1:
            return 0, 2

    if state == 1:
        return np.random.normal(-0.5, 1), 3

    # If a move is made from any other state, nothing happens
    return 0, state

# %%
# Update Q-values


def getMaxQ(Q, state):
    # print(f"getMaxQ => state: {state}, Q: {Q[state]}")
    best_move = np.argmax(Q[state])

    # print(f"best: {best_move}")
    return best_move, Q[state][best_move]


def updateQ(Q, state, action, reward, next_state, alpha=0.1):

    _, next_value = getMaxQ(Q, next_state)

    Q[state][action] = Q[state][action] + alpha * \
        (reward + gamma * next_value - Q[state][action])

    return Q

# %%


def experiment(possible_actions):
    Q = initial_Q(possible_actions)

    ALeft = 0

    left_count = []
    visited = [[0 for _ in possible_actions[state]]
               for state in range(len(possible_actions))]

    state_v = [0 for _ in range(len(possible_actions))]

    for i in range(1, iterations_per_experiments+1):
        state = 0

        while True:
            # increment the state visits
            state_v[state] += 1

            # determine the next action
            action = sample_action(Q, state, 1-1/np.sqrt(state_v[state]))

            # increment left actions
            if state == 0 and action == 0:
                ALeft += 1

            # make transition
            reward, next_state = make_transition(state, action)

            # increment state-action visits
            visited[state][action] += 1

            # determine alpha
            alpha = 1 / np.power(visited[state][action], .8)

            # update Q-value
            Q = updateQ(Q, state, action, reward, next_state, alpha)

            # break if a terminal state is reached
            if is_terminal(next_state):
                break

            state = next_state

        left_count.append(ALeft)

    return left_count


# %%

final_count = [0 for i in range(iterations_per_experiments)]
final_res = [0 for i in range(iterations_per_experiments)]

for i in range(num_experiments):
    current_count = experiment(possible_actions)

    for e in range(iterations_per_experiments):
        final_count[e] = (final_count[e] * i + current_count[e]) / (i+1)
        final_res[e] = final_count[e] / (e+1) * 100

plt.plot(range(len(final_res)), final_res)

# %%

np.argmax([0, 0, 1])
# %%
