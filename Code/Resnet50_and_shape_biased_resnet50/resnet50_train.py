# train resnet50 

from oh_specs import *
import numpy as np
import time
import torch
import torch.nn as nn
import warnings
import argparse
import matplotlib.pyplot as plt

def train():

	####################################################################################################################
    # Model type and logdir. Specify model parameters. 
    ####################################################################################################################

    btch_size     = 16
    n_hidden      = 64
    ns            = range(18) #Tested layer indices 
    use_gpu       = True
    learn_rate    = 1e-6
    n_runs        = 5    
    use_check     = True
    subjs         = range(n_runs)
    tested_layers = 'all_described_layers'
    resnet50s     = ['resnet50_ori',
                	 'resnet50_trained_on_SIN',
                 	 'resnet50_trained_on_SIN_and_IN',
                 	 'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN']
    s_types       = [0,3]  # resnet50 types
    n_batches  	  = 32*4
    n_epochs      = 2000

	####################################################################################################################
    # Training for each backbone model, repeat n_runs times, and save checkpoints
    ####################################################################################################################

    # Run for all 2 different pre-trained models
    for s_type in s_types:

        my_shapenet = ShapeNet(model=resnet50s[s_type], num_classes=2)
        my_shapenet.eval()  # deactivate dropout in resnet50	

        # Train n_runs times and save n_runs decoders
        for run in range(0, n_runs):

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            device = torch.device('cpu')
            if use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                device      = torch.device('cuda')
                my_shapenet  = my_shapenet.to('cuda')
                input_sizes = [get_output_size(my_shapenet, device=device, model_nm=resnet50s[s_type])[n] for n in ns]
                models      = [MyClassifier(input_size=s, n_hidden=n_hidden) for s in input_sizes]
                models      = [model.to('cuda') for model in models]
            else:
                my_shapenet  = my_shapenet.to(device)
                input_sizes = [get_output_size(my_shapenet, device=device, model_nm=resnet50s[s_type])[n] for n in ns]
                models      = [MyClassifier(input_size=s, n_hidden=n_hidden) for s in input_sizes]
                warnings.warn('Pretrained model is on CUDA, should be better to use GPU than CPU')

            crit       = nn.CrossEntropyLoss()
            optims     = [torch.optim.Adam(m.parameters(), lr=learn_rate) for m in models]
            start_epoch= 0

            # if on-going trained model exist, load checkpoint
            model_path  = './SHARED_CODE_logdir/' + resnet50s[s_type] + '/hidden_' + str(n_hidden) + '/lr_' + str(learn_rate) + '/' + tested_layers
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            for i, model in enumerate(models):
                this_model_path = model_path +  '/layer_'+str(ns[i])+'_subject_'+str(run)+'.pt'
                if use_check and os.path.exists(this_model_path):
                    model, optims[i], start_epoch, _ = load_checkpoint(model, optims[i], fname=this_model_path)

            print('\nTraining ' + resnet50s[s_type] + ' ... ')

            # Train for n_epochs
            for e in range(start_epoch, n_epochs):

                e_start      = time.time()
                n_train_errs = [0]*len(ns)
                for b in range(0, n_batches):
                	# generate new dataset for each batch
                    train_i, train_t = make_dataset(btch_size=btch_size, device=device, type='train')

                    with torch.no_grad():
                        input = my_shapenet(train_i, ns)

                    for i, (n, model) in enumerate(zip(ns, models)):
                        if device == 'CUDA':
                            torch.cuda.empty_cache()

                        # Take a batch in the training set and compute the output
                        output = model(input[n])

                        # Compute the loss associated to the model's output
                        loss = crit(output, train_t)
                        n_train_errs[i] += (output.argmax(1) != train_t).sum()

                        # Update the weights
                        model.zero_grad()
                        loss.backward()
                        optims[i].step()

                # Print stuff to monitor the progress on training and validation performance
                n_train_errs = [100 * float(err) / (btch_size * n_batches) for err in n_train_errs]
                n_val_errs   = [0] * len(ns)

                # Print training and testing errors at each epochs
                print('\n\nRun ' + str(run) + ' - Epoch ' + str(e) + '\n')
                for i, n in enumerate(ns):
                    print( '  Layer %2i - train error: %5.2f %% - validation error: %5.2f %% '% (n, n_train_errs[i], n_val_errs[i]))

                # Save the networks every 10 epochs
                e_end = time.time()
                if (e + 1) % 10 == 0:
                    for i, model in enumerate(models):
                        this_model_path = model_path + '/layer_'+str(ns[i])+'_subject_'+str(run)+'.pt'
                        state = {'epoch'     : e+1,
                                 'state_dict': model.state_dict(),
                                 'optimizer' : optims[i].state_dict()}
                        torch.save(state, this_model_path)
                    e_save = time.time()
                    print('Epoch %2i  took %3d min %4.2f sec, and %3d min %4.2f sec to save cp' 
                        % (e,divmod(e_end-e_start,60)[0],divmod(e_end-e_start,60)[1],divmod(e_save-e_end,60)[0],divmod(e_save-e_end,60)[1]))
                        
if __name__ == '__main__':
    print('Train ...')
    train()
