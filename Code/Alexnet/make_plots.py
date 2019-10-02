#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from batch_maker import shapesgen, Lynns_patterns

########################################################################################################################
# Plotting Functions :

def plotFig3a(data, stds, layer_n, show=False, model_path='./'):
    assert len(data) == 21 and len(stds) == 21
    ### Same shape series plot

    fig, ax = plt.subplots()
    index = np.arange(1,6)
    bar_width = 0.21
    opacity = 0.8

    rect0 = ax.bar(0,data[0], 0.33, yerr=stds[0], alpha = opacity, color='darkblue')
    rect1 = ax.bar(index+0*bar_width, data[1:-1:4], bar_width, yerr=stds[1:-1:4], alpha = opacity*1.00, color = ['purple','c', 'olive', 'dimgrey', 'firebrick'])
    rect2 = ax.bar(index+1*bar_width, data[2:-1:4], bar_width, yerr=stds[2:-1:4], alpha = opacity*0.90, color = ['purple','c', 'olive', 'dimgrey', 'firebrick'])
    rect3 = ax.bar(index+2*bar_width, data[3:-1:4], bar_width, yerr=stds[3:-1:4], alpha = opacity*0.75, color = ['purple','c', 'olive', 'dimgrey', 'firebrick'])
    rect4 = ax.bar(index+3*bar_width, data[4:22:4], bar_width, yerr=stds[4:22:4], alpha = opacity*0.60, color = ['purple','c', 'olive', 'dimgrey', 'firebrick'])

    plt.xticks(np.arange(6)+.27, ('No Shape', 'White SQUARE', 'White CIRCLE', 'white HEXAGON', 'Octagon', 'white DIAMOND'))
    plt.xlabel('Shape surrounding the Vernier')
    plt.ylabel('Accuracy')
    ax.plot([-0.3, 5.9], [50, 50], 'k--')  # chance level cashed line
    plt.ylim((40, 100))

    plt.title('Accuracy of decoder number'+str(layer_n+1)+' on Vernier stimuli with same-shape flankers')
    plt.legend(title='Number of flankers:')
    plt.grid(axis='y')

    if not os.path.exists(os.path.join(model_path,'Plots')):
        os.makedirs(os.path.join(model_path,'Plots'))

    if not os.path.exists(os.path.join(model_path,'data_for_plots')):
        os.makedirs(os.path.join(model_path,'data_for_plots'))

    fig.savefig(os.path.join(model_path,'Plots','plotFig3a_decoder_n_' + str(layer_n + 1) + '.png'))
    np.save(os.path.join(model_path,'data_for_plots','plotFig3a_decoder_n_' + str(layer_n + 1)+'_data') , data)
    np.save(os.path.join(model_path,'data_for_plots','plotFig3a_decoder_n_' + str(layer_n + 1)+'_stds') , stds)
    if show:
        plt.show()

########################################################################################################################

def plotFig3b(data, stds, layer_n, lengths, show=False):

    ### Plot performance for all patterns ranked according to n_flanker
    assert len(data) == 88 and len(stds) == 88

    fig, ax = plt.subplots()
    bar_width = 1
    opacity = 0.8

    colors = ['darkblue', 'cornflowerblue', 'mediumseagreen', 'firebrick', 'indianred', 'mediumpurple', 'plum', 'dimgrey']
    c = []
    for i in range(len(lengths)):
        c+=[colors[i]]*lengths[i]
    pos = range(88)

    L = np.array(lengths).cumsum(dtype=float)
    for ix in range(1,len(L)):
        L[ix-1] += L[ix]
        L[ix-1]*=.5
    L-=.5
    L = [-0.3] + L[:-1].tolist()
    for i in pos:
        rect = ax.bar(i, data[i], bar_width, yerr=stds[i], alpha=opacity, ecolor=c[i] , color=c[i])

    bounds = np.array(lengths).cumsum(dtype=int)
    means = [np.array(data[bounds[i]:bounds[i+1]]).mean() for i in range(len(bounds)-1)]
    for i, mu in enumerate(means):
        ax.plot([bounds[i], bounds[i+1]], [mu, mu], 'k:')  # per group mean cashed line


    ax.tick_params(axis='x', which='major', labelsize=7)
    plt.xticks(L, ['None', '1', '3', '5_single', '5_alternate', '7_single', '7_alternate', '7x3'])
    plt.xlabel('Number of Flankers')
    plt.ylabel('Accuracy')
    plt.grid(axis='y')
    plt.ylim((40, 100))
    plt.title('Accuracy of decoder n°' + str(layer_n + 1) + ' on all patterns ranked by number of flankers')
    fig.savefig('Plots/plotFig3b_decoder_n_' + str(layer_n + 1) + '.png')
    np.save('data_for_plots/plotFig3b_decoder_n_' + str(layer_n + 1) + '_data', data)
    np.save('data_for_plots/plotFig3b_decoder_n_' + str(layer_n + 1) + '_stds', stds)
    np.save('data_for_plots/plotFig3b_lengths_for_any', lengths)
    if show:
        plt.show()


if __name__ == '__main__':

    #########################################################################
    # Loading for plotFig3a:
    #########################################################################

    n_hidden = 512
    n_shapes = 5
    n_models = 5
    # Loading for plot 1:
    result_list = []

    for m in range(5):
        path = './SHARED_CODE_logdir/mod_'+str(m)+'_hidden_512/shared_code'

        model_list = []
        # Loading vernier only results
        model_list.append(np.load(path+str([])+'.npy'))

        for shape in range(n_shapes):

            s = [shape+1]
            for i in range(4):

                name = str(s)
                model_list.append(np.load(path+name+'.npy'))
                s += [shape+1, shape+1]
        result_list.append(model_list)

    simple_series = np.array(result_list)

    s_s_avg = simple_series.mean(0)
    s_s_std = simple_series.std(0)

    ## Plotting oncer per Decoder layer
    for i in range(9):
        plotFig3a(s_s_avg[:,i], s_s_std[:, i], i) #for showing plots

    #########################################################################
    # Loading for plotFig3b:
    #########################################################################

    patterns = Lynns_patterns()
    stars = patterns[:8]
    circles = patterns[0:1] + patterns[8:15]
    A_list = []
    B_list = []

    for m in range(5):
        path = './SHARED_CODE_logdir/mod_'+str(m)+'_hidden_512/shared_code'

        star_list = []
        circles_list = []
        no_shape = np.load(path + str([]) + '.npy')
        one_square = np.load(path + str([1]) + '.npy')
        star_list.append(no_shape)
        circles_list.append(no_shape)
        star_list.append(one_square)
        circles_list.append(one_square)

        for p in stars:
            star_list.append(np.load(path + str(p) + '.npy'))
        for p in circles:
            circles_list.append(np.load(path + str(p) + '.npy'))

        A_list.append(star_list)
        B_list.append(circles_list)

    A = np.array(A_list)
    B = np.array(B_list)

    pat = shapesgen(5, False)

    empty = [[]]
    singles = pat[0:35:7]
    threes = pat[1:35:7]
    fives_same = pat[2:35:7]

    l5 = [int(i) for i in np.arange(35)[np.arange(35)%7>2]]
    fives_alt = [pat[i] for i in l5]

    load_list = empty + singles + threes + fives_same + fives_alt
    lengths = [len(empty), len(singles), len(threes), len(fives_same), len(fives_alt)]
    loaded = []

    # vector used for loading 7x alternating patterns
    l7 = [int(i) for i in np.arange(25)[np.arange(25)%5!=0]]

    for m in range(5):
        path = './SHARED_CODE_logdir/mod_'+str(m)+'_hidden_512/shared_code'
        model_list = []

        for n in load_list:
            name = str(n)
            model_list.append(np.load(path + name + '.npy'))

        sevens = np.load(path + '_results.npy')[1:]
        sevens_same = sevens[0:-1:5]
        for x in sevens_same: model_list.append(x)
        sevens_alt = [sevens[i] for i in l7]
        for x in sevens_alt: model_list.append(x)
        loaded.append(model_list)

    lengths += [len(sevens_same), len(sevens_alt), 12]

    loaded = np.array(loaded)
    loaded = np.concatenate((np.array(loaded), A[:,4:10,:], B[:,4:10,:]), axis=1)
    loaded_avg = loaded.mean(0)
    loaded_std = loaded.std(0)

    for i in range(9):
        plotFig3b(loaded_avg[:,i], loaded_std[:,i], i, lengths)
