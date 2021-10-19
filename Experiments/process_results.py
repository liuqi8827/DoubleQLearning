import numpy as np
import os
import csv

def process_results(results):
    lengths = np.array(results[0])
    mean_lengths, std_lengths = get_stats(lengths)
    
    returns = np.array(results[1])
    mean_returns, std_returns = get_stats(returns)

    return (mean_lengths, std_lengths), (mean_returns, std_returns)


def get_stats(data):
    data = np.array(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return mean, std


def save_results(single_results, double_results, save_dir='Results'):
    os.makedirs(save_dir, exist_ok=True)
    write_result(single_results[0], 'Single Lengths', save_dir=save_dir)
    write_result(single_results[1], 'Single Returns', save_dir=save_dir)
    write_result(double_results[0], 'Double Lengths', save_dir=save_dir)
    write_result(double_results[1], 'Double Returns', save_dir=save_dir)
    return


def write_result(results, header, save_dir='Results', file_id=''):
    file_name = header + file_id + '.csv'
    file_path = os.path.join(save_dir, file_name)

    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        header = [header] + [i for i in range(len(results[0]))]
        writer.writerow(header)

        for j in range(len(results)):
            row = [j] + list(results[j])
            writer.writerow(row)
        mean, std = get_stats(results)
        mean_row = ['Mean'] + mean.tolist()
        std_row = ['Std. Dev.'] + std.tolist()
        writer.writerow(mean_row)
        writer.writerow(std_row)
    return
