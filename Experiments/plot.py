import matplotlib.pyplot as plt
from process_results import process_results
import os


def plot_results(single_results, double_results, save_dir=None, file_id='', file_type='.png', show=False):
    # A comparison of double_Q vs single_Q over all runs
    # A comparison of the first half of single against itself and double against itself
    # A comparison of the first half of double and single vs the second half

    (mean_single_lengths, std_single_lengths), (mean_single_returns, std_single_returns) = process_results(single_results)
    (mean_double_lengths, std_double_lengths), (mean_double_returns, std_double_returns) = process_results(double_results)

    # So 3x2 plots
    fig = plt.figure(figsize=(20, 5))
    # fig.suptitle("Results")

    # plt.text("Episode Returns")
    # plt.text("Episode Lengths")
    # A comparison of episode returns for single_Q vs double_Q over all runs
    fig.add_subplot(121)
    plot_metric("Episode Returns", mean_single_returns, std_single_returns, mean_double_returns, std_double_returns)

    # A Comparison of episode lengths for single_Q vs double_Q over all runs
    fig.add_subplot(122)
    plot_metric("Episode Lengths", mean_single_lengths, std_single_lengths, mean_double_lengths, std_double_lengths)

 

    # # A comparison of the episode returns for different random seeds for single Q 
    # fig.add_subplot(323)
    # plot_metric(single_q_returns[:half], single_q_returns[half:])

    # # A comparison of the episode lengths for different random seeds for single Q
    # fig.add_subplot(324)
    # plot_metric(single_q_lengths[:half], single_q_lengths[half:])

    # # A comparison of the episode returns for different random seeds for double Q 
    # fig.add_subplot(325)
    # plot_metric(double_q_returns[:half], double_q_returns[half:])

    # # A comparison of the episode lengths for different random seeds for double Q 
    # fig.add_subplot(326)
    # plot_metric(double_q_lengths[:half], double_q_lengths[half:])
    
    if save_dir:
        file_path = save_dir + file_type
        plt.savefig(file_path)
    
    plt.show()
    
    return


def plot_metric(title, single_mean, single_std, double_mean, double_std, save_dir=None, file_type='.png', save=False, show=False):

    plt.plot(single_mean)
    plt.fill_between(range(len(single_mean)), single_mean - single_std, single_mean + single_std, alpha=0.3)
    plt.plot(double_mean)
    plt.fill_between(range(len(single_mean)), double_mean - double_std, double_mean + double_std, alpha=0.3)
    plt.title(title)
    plt.legend(['Single Q-Learning', "Double Q-learning"])

    if save_dir:
        filename = title + file_type
        file_path = os.path.join(save_dir, filename)
        plt.savefig(file_path)

    if show:
        plt.show()

    return