from occlusion_specs import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
from oh_specs import *
import os
fixed_noise = np.random.normal(0, 0.1, size=(1, 227, 227, 3))

# Occlusion map
def occlusion(stimMatrix, test_type, offset_type, dilation, models, backbone_model, model_nm='model', occlu_path='./SHARED_CODE_occlusion_maps/'):
    '''
    Occlusion sentivity map
    - predition() function should return B x 2 tensor, which is the score of L/R
    :return: Occlusion sensitivity map images (saved in save_path directly)
    '''

    n_classifier = 18
    ratios = [0, 0, 0, 1]  # ratios : 0 - vernier alone; 1- shapes alone; 2- Vernier ext; 3-vernier inside shape
    this_offset = offset_type
    
    print('Creating ' + test_type + ' dataset ...')
    imgs, label, grid = make_occlusion_dataset(shapeMatrix, ratios, frame_work='pytorch', dilation=dilation, offset=this_offset)
    # change to tensor
    imgs = torch.from_numpy(imgs)
    
    # insidef the occlusion map has already been computed, load it, otherwise, compute.
    map_name = os.path.join(occlu_path, 'occlu_' + str(test_type) + '_' + str(dilation) + '_' + str(offset_type) + '_'+ model_nm+'_modelID_' + str(model_ID))
    if os.path.exists(map_name + '.npy'):
        print('occlumap found, loading for stimulus ' + str(stimMatrix))
        occlu_map = np.load(map_name + '.npy')

    else:
        print('No occlumap found, computing for stimulus ' + str(stimMatrix))
        
        r = len(grid[0])
        c = len(grid[1])

        occlu_map = np.zeros((n_classifier, r, c)) # occlu_map: n_classifier x n_pos x r x c
        target_vector = 2*(np.hstack([1-label[0], label[0]])-0.5) # vector of size [n_imgs, 2], label 0 -> [1,-1], label 1 -> [-1, 1]
        
        print('Predicting '+ str(test_type) + ' occluded images ...')
        occlu_list = prediction(imgs, models, backbone_model)  # occlu_list: total_imgs[0] x n_classifier x 2

        for c in range(n_classifier):
            pred = occlu_list[0, c, :]  # output on image without occlusion
            scores = np.sum(target_vector*(occlu_list[1:, c, :] - np.tile(pred, [occlu_list.shape[0]-1, 1])), axis=-1)  # score taking into account wether the occlusion moves the result towards the correct or incorrect response
            occlu_map[c, :len(grid[0]), :len(grid[1])] = np.reshape(scores, (len(grid[0]), len(grid[1])))

    # plot and save figures + occlumap
    plt.figure(figsize=(20, 20))
    for c in range(n_classifier):
        plt.subplot(4, 5, c+1)
        max_score = 1
        patch_size = 6
        padding = patch_size//2
        stim_img = imgs[0,0,padding:-padding,padding:-padding]
        occlu_img = np.zeros_like(stim_img)
        print(c,stim_img.shape, occlu_img.shape)
        occlu_img[:occlu_map[c,:,:].shape[0],:occlu_map[c,:,:].shape[1]] = occlu_map[c,:,:]
        plt.imshow(10*occlu_img, cmap='RdBu', norm=mplt.colors.Normalize(vmin=-max_score, vmax=max_score))
        plt.colorbar()
        plt.imshow(stim_img, cmap='Greys', norm=mplt.colors.Normalize(vmin=0, vmax=max_score), alpha=0.1)
        plt.axis('off')
        plt.title('Layer' + str(c+1))
    plt.savefig(map_name + '.png', transparent=True, bbox_inches='tight', pad_inches=0)
    np.save(map_name, occlu_map)

    print('Occlusion sensitivity map for ' + str(test_type) + ' is saved to: ' + occlu_path)


def prediction(imgs, models, backbone_model):

    btch_size = 128
    occlu_list = np.zeros((imgs.shape[0], 18, 2))
    it = 0
    for b in range(0, imgs.shape[0], btch_size):
        print('\rBatch {}/{}'.format(it, len(range(0, imgs.shape[0], btch_size))))
        this_batch = imgs[b:min(b+btch_size,imgs.shape[0]),:,:,:]
        occlu_list[b:min(b+btch_size, imgs.shape[0]), :, :] = get_batch_outputs_pytorch(this_batch, models, backbone_model)
        it += 1
    return occlu_list


if __name__ == '__main__':

    print('OCCLUSION ...')

    occlu_path = './SHARED_CODE_occlusion_maps/'
    if not os.path.exists(occlu_path):
        os.makedirs(occlu_path, exist_ok=True)
    
    # Model parameters
    resnet50s  = ['resnet50_ori',
                  'resnet50_trained_on_SIN',
                  'resnet50_trained_on_SIN_and_IN',
                  'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN']
    model_types   = [0,3]
    tested_layers = 'all_described_layers'
    ns            = range(18)
    n_hidden      = 64
    use_gpu       = True
    use_check     = True
    learn_rate    = 1e-6

    # Set dataset configurations
    test_types = ['00ver', '01square', '07squares', '07square&stars', '01circle', '07circles', '07circle&stars', '01hexagon', '07hexagons', '07hexagon&stars', '21squares_stars_uncrowd', '21squares_stars_crowd']
    shapeMatrices = [[], [1], [1, 1, 1, 1, 1, 1, 1], [6, 1, 6, 1, 6, 1, 6], [2], [2, 2, 2, 2, 2, 2, 2], [6, 2, 6, 2, 6, 2, 6], [3], [3, 3, 3, 3, 3, 3, 3], [6, 3, 6, 3, 6, 3, 6], [[6, 1, 6, 1, 6, 1, 6], [6, 1, 6, 1, 6, 1, 6], [6, 1, 6, 1, 6, 1, 6]], [[1, 6, 1, 6, 1, 6, 1], [6, 1, 6, 1, 6, 1, 6], [1, 6, 1, 6, 1, 6, 1]]]
    
    for model_type in model_types:
        
        # Load trained models
        my_shapenet = ShapeNet(model= resnet50s[model_type], num_classes=2)
        my_shapenet.eval()  # deactivate dropout in resnet50
       
        device = torch.device('cpu')
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            device = torch.device('cuda')
            my_shapenet = my_shapenet.to('cuda')
            input_sizes = [get_output_size(my_shapenet, device=device, model_nm=resnet50s[model_type])[n] for n in ns]
            models = [MyClassifier(input_size=s, n_hidden=n_hidden) for s in input_sizes]
            models = [model.to('cuda') for model in models]
        else:
            my_shapenet = my_shapenet.to(device)
            input_sizes = [get_output_size(my_shapenet, device=device, model_nm=resnet50s[model_type])[n] for n in ns]
            models = [MyClassifier(input_size=s, n_hidden=n_hidden) for s in input_sizes]
            warnings.warn('Pretrained model is on CUDA, should be better to use GPU than CPU')

        # Load saved checkpoints
        model_path  = './SHARED_CODE_logdir/' + resnet50s[model_type] + '/hidden_' + str(n_hidden) + '/lr_' + str(learn_rate) + '/' + tested_layers
    
        for model_ID in range(5):
        
            for i, model in enumerate(models):
                this_model_path = model_path + '/layer_' + str(ns[i]) + '_subject_' + str(model_ID) + '.pt'
                model, _, _, _= load_checkpoint(model, fname=this_model_path)
            
            for i, shapeMatrix in enumerate(shapeMatrices):
                for offset_type in [0, 1]:  # left vernier, right vernier
                    print('\n\nMODEL %1i processing stimulus %s with offset %1i' % (model_ID, test_types[i], offset_type))
                    occlusion(shapeMatrix, test_types[i], offset_type, dilation=1, models=models, backbone_model=my_shapenet, model_nm=resnet50s[model_type])

    print('We are done.')