import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from ray.tune.visual_utils import load_results_to_df

files = [
    "mnist_inv1x1_grid",
    "mnist_conv3x3_2",
    "mnist_inv1x1_random2",
    "mnist_inv1x1_random_impulse2",
    "mnist_inv1x1_random_smooth",
    "mnist_inv1x1_dct",
    "mnist_inv3x3_1x1",
    "mnist_inv3x3",
    "mnist_conv3x3_wide",
]

if __name__ == '__main__':
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Computer Modern Roman'

    basedir = '~/mario/scratch/ray_results'
    GOOD_FIELDS = ['lr', 'momentum', 'wd', 'std', 'mean_accuracy']
    bins = np.linspace(90, 100, 21)
    bw = (bins[1]-bins[0])
    x = bins[:-1] + bw

    for f in files:
        plt.cla()
        results_dir = os.path.expanduser(os.path.join(basedir, f))
        df1 = load_results_to_df(results_dir)
        df1 = df1[GOOD_FIELDS]
        df1 = df1.dropna()
        cnt1, _ = np.histogram(100*df1.mean_accuracy[df1.mean_accuracy > 0.9], bins=bins, density=True)
        plt.bar(x, cnt1, width=bw, color='b', alpha=0.5, edgecolor='k')
        plt.xlabel('Accuracy (\%)')
        plt.ylabel(r'Normalized Counts')
        plt.ylim(0, 0.8)
        plt.gca().set_position((0.15, 0.15, 0.75, 0.75))
        plt.savefig(f + '.pdf', dpi=300)
        print('Saved pdf for {}'.format(f))
