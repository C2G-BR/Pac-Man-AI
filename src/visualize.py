import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from os.path import join

def visualize() -> None:
    """Visualize Reward per Episode

    Visualize the reward per Episode for all runs. This requires the csv data as
    it can be downloaded from tensorboard. The file name should not be changed
    since the name of the run will be used inside the legend. The files must be
    placed inside the 'data' folder. The image will be generated inside the
    'img' folder.

    A sliding window with size N=100 will be applied to reduce the noise.
    """
    directory = join(os.getcwd(), 'data')
    run_dfs = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            df = pd.read_csv(join(directory, filename))
            name = filename.split('-')[1]
            run_dfs[name] = df

    plt.figure()
    for name, df in run_dfs.items():
        N = 100
        # Apply sliding windows with size N.
        mean = np.convolve(df['Value'], np.ones(N)/N, mode='valid')
        plt.plot(df['Step'].iloc[N-1:], mean, label=name)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('img/rewards_per_episode_windows.png')

if __name__ == '__main__':
    visualize()