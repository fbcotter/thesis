import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
from ray.tune.visual_utils import load_results_to_df
pd.options.display.max_rows = 999
types = ['gainA', 'gainB', 'gainC', 'gainD', 'gainE', 'gainF', 'gainAB',
         'gainBC', 'gainCD', 'gainDE', 'gainAC', 'gainBD', 'gainCE']
cifar100_files = ["/home/fbc23/mario/scratch/ray_results/gainlayer_cifar10/",
                  "/home/fbc23/mario/scratch/ray_results/gainlayer_dwt_cifar10/"]

if __name__ == '__main__':
    mpl.rcParams['font.size'] = 16
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Computer Modern Roman'

    assert os.path.exists(cifar100_files[0])
    assert os.path.exists(cifar100_files[1])
    df1 = load_results_to_df(cifar100_files[0])
    df2 = load_results_to_df(cifar100_files[1])
    GOOD_FIELDS = ['type', 'mean_accuracy']
    df1 = df1[GOOD_FIELDS]
    df1 = df1.dropna()
    df2 = df2[GOOD_FIELDS]
    df2 = df2.dropna()
    group1 = df1.groupby(['type'])
    group2 = df2.groupby(['type'])
    vals1 = group1['mean_accuracy'].agg(['mean', 'std'])
    vals2 = group2['mean_accuracy'].agg(['mean', 'std'])
    ref = pd.concat([df1[df1.type == 'ref'], df2[df2.type=='ref']])

    plt.figure(figsize=(7,8))
    w = 0.2
    space = 0.1
    lw = 1
    plt.barh(-1, ref.mean_accuracy.mean(), xerr=ref.mean_accuracy.std(),
             edgecolor='k', color='k', alpha=0.2, align='center', height=2*w,
             error_kw={'ecolor': 'k', 'capsize': 0, 'elinewidth': lw})
    plt.barh(np.arange(len(types))-w/2-space/2, vals2.loc[types,'mean'],
             xerr=vals2.loc[types,'std'], edgecolor='r',
             color='r', alpha=0.2, align='center', height=w,
             error_kw={'ecolor': 'r', 'capsize': 0, 'elinewidth': lw})
    plt.barh(np.arange(len(types))+w/2+space/2, vals1.loc[types,'mean'],
             xerr=vals1.loc[types,'std'], edgecolor='b',
             color='b', alpha=0.2, align='center', height=w,
             error_kw={'ecolor': 'b', 'capsize': 0, 'elinewidth': lw})

    plt.xlim(90, 95)
    plt.yticks(range(-1, vals1.shape[0]), labels=['ref'] + types)
    #plt.plot(range(5))
    plt.gca().invert_yaxis()
    plt.axvline(ref.mean_accuracy.mean(), color='k', ls='--', lw=1)
    plt.xlabel('Accuracy (\%)')
    plt.show()
