import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_result(exp_name):
    path = os.path.join('useful', exp_name, 'log.txt')
    return pd.read_csv(path, sep='\t')


def add_plot(data, var_name, label=''):
    sns.set(style="darkgrid", font_scale=0.5)
    plt.plot(data['Timestep'], data[var_name], label=label, lw=1, alpha=0.5)


def main():
    if not os.path.exists('results'):
        os.makedirs('results')


if __name__ == '__main__':
    main()
