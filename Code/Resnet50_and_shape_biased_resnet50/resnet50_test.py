# test resnet50 and plot figure 3

from oh_specs import *
import numpy as np
import time
import torch
import torch.nn as nn
import warnings
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm


def test():

	####################################################################################################################
    # Model type and logdir. Specify model parameters. 
    ####################################################################################################################

    btch_size     = 100
    n_hidden      = 64
    ns            = range(18) #Tested layer indices 
    use_gpu       = True
    learn_rate    = 1e-6
    n_runs        = 5
    subjs         = range(n_runs)
    tested_layers = 'all_described_layers'
    resnet50s     = ['resnet50_ori',
                	 'resnet50_trained_on_SIN',
                 	 'resnet50_trained_on_SIN_and_IN',
                 	 'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN']
    s_types       = [3,0]  # resnet50 types
    

    for s_type in s_types:
        print('Testing model ' + resnet50s[s_type])

        e_start = time.time()

        my_shapenet = ShapeNet(model=resnet50s[s_type], num_classes=2)
        my_shapenet.eval()  # deactivate dropout in resnet50

        device = torch.device('cpu')
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            device = torch.device('cuda')
            my_shapenet = my_shapenet.to('cuda')
            input_sizes = [get_output_size(my_shapenet, device=device)[n] for n in ns]
            models = [MyClassifier(input_size=s, n_hidden=n_hidden) for s in input_sizes]
            models = [model.to('cuda') for model in models]
        else:
            my_shapenet = my_shapenet.to(device)
            input_sizes = [get_output_size(my_shapenet, device=device)[n] for n in ns]
            models = [MyClassifier(input_size=s, n_hidden=n_hidden) for s in input_sizes]
            warnings.warn('Pretrained model is on CUDA, should be better to use GPU than CPU')

        # load saved checkpoint and test configurations to plot as in figure 3
        model_path = './SHARED_CODE_logdir/' + resnet50s[s_type] + '/hidden_' + str(n_hidden) + '/lr_' + str(learn_rate) + '/' + tested_layers
        shapeMatrix = [[],[1],[1]*3,[1]*5,[1]*7,[2],[2]*3,[2]*5,[2]*7,
                          [3],[3]*3,[3]*5,[3]*7,[4],[4]*3,[4]*5,[4]*7,
                          [5],[5]*3,[5]*5,[5]*7]
        test_types  = ['ver','1sq','3sq','5sq','7sq','1cc','3cc','5cc','7cc','1hg','3hg','5hg','7hg',
                       '1og','3og','5og','7og','1dm','3dm','5dm','7dm']
        test_models(my_shapenet, models, n_blocks=64, n_samples=btch_size, device=device, model_path=model_path, ns=ns, subjs=subjs, shapeMatrix=shapeMatrix, test_types=test_types)

        e_end = time.time()
        print('Model %s  took %3d min %4.2f sec' % (resnet50s[s_type], divmod(e_end - e_start, 60)[0], divmod(e_end - e_start, 60)[1]))


if __name__ == '__main__':
    print('TEST ...')
    test()
