from Algorithms.q_learning import SingleQLearning, DoubleQLearning
from Algorithms.policy import EpsilonGreedyPolicy
from typing import List
import numpy as np
from Experiments.process_results import save_results
from Experiments.plot import plot_results
import os
from tqdm import tqdm


def running_mean(vals, n=1):
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n


def experiment(env_name, env, params: dict, seeds : List[int], default_dir='Results'):
    
    """
    The full loop for an experiment. 
    Runs the experiment across multiple seeds, saves results, and plots them.

    Args:
        env_name: name of the current environment (string)
        env: An OpenAI env
        params: a dict containing the parameters (with default values) 
            epsilon: 0.1, 
            num_episodes: (1000), 
            discount_factor: 1.0, 
            alpha : 0.1
        seeds: A list of ints, used as seeds for np.random.seed()

    Returns: 
        Episode Lengths, Episode Returns
    """


    print("Starting experiment")
    
    # Get results for test
    single_results, double_results = run_experiment(env, seeds=seeds, params=params)

    # Save Results to csv
    str_params = ''
    if not params is None:
        for key, value in params.items():
            str_params += key + '=' + str(value) + '_'
    save_dir = os.path.join(default_dir, env_name, str_params)
    save_results(single_results, double_results, save_dir=save_dir)

    # Plot results
    plot_results(single_results, double_results, save_dir=save_dir)

    #Close environment to prevent issues
    env.close()
    print("\n\n")
    
    return


def run_experiment(env, seeds : List[int] = None, params : dict = None, show_episodes=False):
    """
    Runs an experiment across multiple seeds

    args:


    returns: 
        Episode Lengths, Episode Returns
    """

    # If no seeds are provided, simply take the indices of the runs
    if seeds is None:
        seeds = [i for i in range(10)]

    # Obtain results for each random seed
    single_lengths = []
    single_returns = []
    double_lengths = []
    double_returns = []
    
    for i in tqdm(seeds):
        np.random.seed(i)
        single_result, double_result = single_run(env, params=params, show_episodes=show_episodes)
        single_lengths.append(single_result[0])
        single_returns.append(single_result[1])
        double_lengths.append(double_result[0])
        double_returns.append(double_result[1])
        

    return (single_lengths, single_returns), (double_lengths, double_returns)


def single_run(env, params : dict = None, show_episodes=False):
    """
    A single training run for a single env

    returns: Episode Lengths, Episode Returns
    """
    if params is None:
        params = {}

    epsilon = params.setdefault('epsilon', 0.1)
    num_episodes = params.setdefault('num_episodes', 1000)
    discount_factor = params.setdefault('discount_factor', 1)
    alpha = params.setdefault('alpha', 0.1)
    

    # Obtain environment variables
    num_states = env.nS
    num_actions = env.nA

    # For a single run, obtain all metrics and return
    Q = np.zeros((num_states, num_actions))

    # Single Q
    single_policy = EpsilonGreedyPolicy(epsilon, Q)
    single_q = SingleQLearning()
    single_q_vals, single_q_results = \
        single_q.train(env, single_policy, num_episodes, discount_factor=discount_factor, alpha=alpha, show_episodes=show_episodes)


    # Double Q
    Q_a = np.zeros((num_states, num_actions))
    Q_b = np.zeros((num_states, num_actions))
    double_policy = EpsilonGreedyPolicy(epsilon, Q_a, Q_b)
    double_q = DoubleQLearning()
    double_q_vals, double_q_results = \
        double_q.train(env, double_policy, num_episodes, discount_factor=discount_factor, alpha=alpha, show_episodes=show_episodes)

    return (single_q_results, double_q_results)

