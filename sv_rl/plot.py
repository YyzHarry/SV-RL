import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_data(data, value="MeanReward100Episodes", save_name="results", title="Pong"):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    plt.figure()
    # sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(data=data, time="Timestep", value=value, unit="Unit", condition="Condition", lw=2, alpha=1)
    plt.xlabel('Time step', fontsize=20)
    plt.ylabel('Average score', fontsize=20)
    plt.legend(loc='best', fontsize=15).draggable()
    plt.title(title, fontsize=20)
    ax = plt.axes()
    ax.tick_params(direction='in')
    plt.grid()
    plt.tick_params(labelsize=15)
    plt.show()
    plt.savefig(
        os.path.join('results/final', save_name + '.pdf'),
        bbox_inches='tight',
        transparent=True,
        pad_inches=0.1
    )


def get_datasets(fpath, condition=None):
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'log.txt' in files:
            log_path = os.path.join(root, 'log.txt')
            experiment_data = pd.read_csv(log_path, sep='\t')

            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
                )        
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                condition
                )

            datasets.append(experiment_data)
            unit += 1

    return datasets


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--value', default='MeanReward100Episodes', nargs='*')
    parser.add_argument('--save_name', type=str, default='results')
    parser.add_argument('--title', type=str, default='Pong')
    args = parser.parse_args()
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    data = []
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            data += get_datasets(logdir, legend_title)
    else:
        for logdir in args.logdir:
            data += get_datasets(logdir)

    if isinstance(args.value, list):
        values = args.value
    else:
        values = [args.value]
    for value in values:
        plot_data(data, value=value, save_name=args.save_name, title=args.title)


if __name__ == "__main__":
    main()
