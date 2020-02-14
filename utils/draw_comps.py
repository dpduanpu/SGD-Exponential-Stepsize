"""
Script for drawing comparison between different optimization methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import os
import argparse

good_colors = [
    [0, 0, 1.0000],
    [1.0000, 0, 0],
    [0.1, 0.9500, 0.1],
    [0, 0, 0.1724],
    [1.0000, 0.1034, 0.7241],
    [1.0000, 0.8276, 0],
    [0, 0.3448, 0],
    [0.5172, 0.5172, 1.0000],
    [0.6207, 0.3103, 0.2759],
    [0, 1.0000, 0.7586]
]

def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m, se = np.mean(data), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def readFile(folder, fileName, stat_names):
    """From folder read fileName, which contains fields in stat_names."""
    fullFileName = os.path.join(folder, fileName)
    results = {}
    num_reps = 0 # Number of independent runs.
    with open(fullFileName, 'r') as f:
        counter = 0
        for line in f:
            if not line.startswith('['):
                continue
            stat_name = stat_names[counter]
            value_array = np.array(eval(line))
            results[stat_name] = (results.get(stat_name, np.zeros(len(value_array)))
                                 + value_array)
            counter = (counter + 1) % 4
            num_reps += 1 if counter == 0 else 0

    for stat_name in stat_names:
        results[stat_name] = results[stat_name] / num_reps

    return results


def draw_comps(folder):
    linewidth = 3
    fontsize = 14

    stat_names = ['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy']
    labels = {'Adam':'Adam', 'SGD':'SGD constant', 'SGD_1t_Decay':'O(1/t)',
              'SGD_1sqrt_Decay':'O(1/sqrt(t))', 'SGD_Exp_Decay':'O(alpha^t)',
              'SGD_ReduceLROnPlateau':'ReduceLROnPlateau'}
    colors = {'Adam':0, 'SGD_1sqrt_Decay':1, 'SGD_1t_Decay':2, 'SGD_Exp_Decay':3, 
              'SGD':4, 'SGD_ReduceLROnPlateau':5}

    files = os.listdir(folder)
    for fileName in files:
        stats = readFile(folder, fileName, stat_names)
        method = fileName[fileName.find('_') + 1 : fileName.find('_Eta0')]
        if method == 'SGD_Stage_Decay':
            miles = fileName[fileName.find('Milestones_') + 11 : fileName.find('_Epoch')]
            num_miles = len(miles.split('_'))
            label = ('Stagewise %d milestone%s' % (num_miles, 's' if num_miles > 1 else ''))
            color = good_colors[len(colors) + num_miles]
        else:
            label = labels[method]
            color = good_colors[colors[method]]            

        for j, stat_name in enumerate(stat_names):
            plt.figure(j)
            plt.plot(stats[stat_name], linewidth=linewidth,
                     label=label, color=color)

    for i, stat_name in enumerate(stat_names):
        plt.figure(i)
        plt.xlabel('Number of epochs', fontsize=fontsize)
        plt.ylabel(stat_name, fontsize=fontsize)
        legend_loc = 'upper right'
        if stat_name.find('Accuracy') != -1:
            legend_loc = 'lower right'
        plt.legend(loc=legend_loc, fontsize=fontsize)

    plt.show()

def load_args():
    parser = argparse.ArgumentParser(description='Draw figures.')

    parser.add_argument('--folder', type=str, default='./logs/tests',
                        help='log folder path')

    return parser.parse_args()


args = load_args()
draw_comps(args.folder)