from Algorithms.single_q_learning import SingleQLearning
from Algorithms.double_q_learning import DoubleQLearning
from typing import List
import numpy as np
from process_results import save_results
from plot import plot_results
import os
from tqdm import tqdm

def running_mean(vals, n=1):
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n


def custom_experiment(env_name, env, params: dict, seeds : List[int], default_dir='Results'):
    
    print("Starting custom experiment")
    
    # Get results for test
    single_results, double_results = run_experiment(env, policy=None, seeds=seeds, params=params)

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


def run_experiment(env, policy=None, seeds : List[int] = None, params : dict = None, show_episodes=False):
   
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
        single_result, double_result = single_run(env, policy=policy, params=params, show_episodes=show_episodes)
        single_lengths.append(single_result[0])
        single_returns.append(single_result[1])
        double_lengths.append(double_result[0])
        double_returns.append(double_result[1])
        

    return (single_lengths, single_returns), (double_lengths, double_returns)


def single_run(env, policy=None, params : dict = None, show_episodes=False):
    if params is None:
        params = {}

    epsilon = params.setdefault('epsilon', 0.1)
    num_episodes = params.setdefault('num_episodes', 1000)
    discount_factor = params.setdefault('discount_factor', 1)
    alpha = params.setdefault('alpha', 0.1)
    
    # For a single run, obtain all metrics and return
    single_q_vals, single_q_results = single_q(env, policy=policy, epsilon=epsilon, num_episodes=num_episodes, \
        discount_factor=discount_factor, alpha=alpha, show_episodes=show_episodes)
    double_q_vals, double_q_results = double_q(env, policy=policy, epsilon=epsilon, num_episodes=num_episodes, \
        discount_factor=discount_factor, alpha=alpha, show_episodes=show_episodes)
    return (single_q_results, double_q_results)

        
def single_q(env, policy=None, epsilon=0.1, num_episodes=1000, discount_factor=1, alpha=0.1, show_episodes=False):
        
    if policy is None:
        Q = {}
        policy = SingleQLearning.EpsilonGreedyPolicy(Q, epsilon=epsilon)

    Q_values, (episode_lengths, episode_returns) = \
        SingleQLearning.single_q_learning(env, policy, num_episodes, Q=None,\
            discount_factor=discount_factor, alpha=alpha, show_episodes=show_episodes)
    return Q_values, (episode_lengths, episode_returns)
    

def double_q(env, policy=None, epsilon=0.1, num_episodes=1000, discount_factor=1, alpha=0.1, show_episodes=False):

    if policy is None:
        Q_a = {}
        Q_b = {}
        policy = DoubleQLearning.EpsilonGreedyPolicy(Q_a, Q_b, epsilon=epsilon)

    Q_values, (episode_lengths, episode_returns) = \
        DoubleQLearning.double_q_learning(env, policy, num_episodes, Q_a=None, Q_b=None,\
            discount_factor=discount_factor, alpha=alpha, show_episodes=show_episodes)
    return  Q_values, (episode_lengths, episode_returns)


