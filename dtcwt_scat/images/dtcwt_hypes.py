import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from ray.tune.visual_utils import load_results_to_df
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import floor, ceil

files = [
    "cifar100_scat_options",
]

if __name__ == '__main__':
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Computer Modern Roman'

    basedir = '~/temp'
    GOOD_FIELDS = ['biort', 'mode', 'magbias', 'mean_accuracy']

    for f in files:
        plt.cla()
        results_dir = os.path.expanduser(os.path.join(basedir, f))
        df1 = load_results_to_df(results_dir)
        df1 = df1[GOOD_FIELDS]
        df1 = df1.dropna()

        averages = df1.groupby(['biort', 'mode', 'magbias',]).mean()
        averages = averages.reset_index()
        zero = averages[averages['mode'] == 'zero']
        zero = zero.sort_values(by=['biort', 'magbias'])['mean_accuracy'].values
        zero = zero.reshape(3,4)
        symm = averages[averages['mode'] == 'symmetric']
        symm = symm.sort_values(by=['biort', 'magbias'])['mean_accuracy'].values
        symm = symm.reshape(3,4)

        fig = plt.figure(figsize=(9,3))
        #  fig.suptitle('CIFAR-100', x=0.54)
        ax = plt.subplot(111)

        vmin = floor(min(zero.min(), symm.min()))
        vmax = ceil(max(zero.max(), symm.max()))
        img = ax.imshow(zero, vmin=vmin, vmax=vmax, cmap='viridis')
        ax.set_yticks([0,1,2])
        ax.set_yticklabels(['near\_sym\_a', 'near\_sym\_b', 'near\_symb\_b\_bp'])
        ax.set_ylabel('wave')
        ax.set_title('zero pad')
        ax.set_xlabel(r'smooth factor $b$')
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels([0, r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        divider.set_position([0.2, 0.16, 0.75, 0.75])

        ax2 = divider.append_axes("right", size="100%", pad=0.05)
        ax2.set_title('symmetric pad')
        ax2.set_xlabel(r'smooth factor $b$')
        ax2.set_yticks([])
        ax2.set_xticks([0,1,2,3])
        ax2.set_xticklabels([0, r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
        im = ax2.imshow(symm, vmin=vmin, vmax=vmax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_ticks(list(range(vmin, vmax+1)))
        plt.savefig(f + '.pdf', dpi=300)
        print('Saved pdf for {}'.format(f))
