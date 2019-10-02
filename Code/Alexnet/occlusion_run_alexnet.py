from occlusion_specs import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
from alexnet_with_vernier_decoders import get_batch_outputs
import os
fixed_noise = np.random.normal(0, 0.1, size=(1, 227, 227, 3))

# Occlusion map
def occlusion(stimMatrix, test_type, offset_type, dilation, model_ID=4):
    '''
    Occlusion sentivity map
    - predition() function should return B x 2 tensor, which is the score of L/R
    :return: Occlusion sensitivity map images (saved in save_path directly)
    '''

    n_classifier = 9
    ratios = [0, 0, 0, 1]  # ratios : 0 - vernier alone; 1- shapes alone; 2- Vernier ext; 3-vernier inside shape
    this_offset = offset_type

    print('Creating ' + test_type + ' dataset ...')
    imgs, label, grid = make_occlusion_dataset(shapeMatrix, ratios, dilation=dilation, offset=this_offset)

    # if the occlusion map has already been computed, load it, otherwise, compute.
    map_name = os.path.join(occlu_path, 'occlu_' + str(test_type) + '_' + str(dilation) + '_' + str(offset_type) + '_modelID_' + str(model_ID))
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
        occlu_list = prediction(imgs, model_ID)  # occlu_list: total_imgs[0] x n_classifier x 2

        for c in range(n_classifier):
            pred = occlu_list[0, c, :]  # output on image without occlusion
            scores = np.sum(target_vector*(occlu_list[1:, c, :] - np.tile(pred, [occlu_list.shape[0]-1, 1])), axis=-1)  # score taking into account wether the occlusion moves the result towards the correct or incorrect response
            occlu_map[c, :len(grid[0]), :len(grid[1])] = np.reshape(scores, (len(grid[0]), len(grid[1])))

    # plot and save figures + occlumap
    plt.figure(figsize=(20, 20))
    for c in range(n_classifier):
        plt.subplot(3, 3, c+1)
        max_score = 1
        patch_size = 6
        padding = patch_size//2
        stim_img = imgs[0,padding:-padding,padding:-padding,0]
        occlu_img = np.zeros_like(stim_img)
        occlu_img[:occlu_map[c,:,:].shape[0],:occlu_map[c,:,:].shape[1]] = occlu_map[c,:,:]
        plt.imshow(10*occlu_img, cmap='RdBu', norm=mplt.colors.Normalize(vmin=-max_score, vmax=max_score))
        plt.colorbar()
        plt.imshow(stim_img, cmap='Greys', norm=mplt.colors.Normalize(vmin=0, vmax=max_score), alpha=0.1)
        plt.axis('off')
        plt.title('Layer' + str(c+1))
    plt.savefig(map_name + '.png', transparent=True, bbox_inches='tight', pad_inches=0)
    np.save(map_name, occlu_map)

    print('Occlusion sensitivity map for ' + str(test_type) + ' is saved to: ' + occlu_path)


def prediction(imgs, model_ID=4):

    btch_size = 512
    occlu_list = np.zeros((imgs.shape[0], 9, 2))
    it = 0
    for b in range(0, imgs.shape[0], btch_size):
        print('\rBatch {}/{}'.format(it, len(range(0, imgs.shape[0], btch_size))))
        this_batch = imgs[b:min(b+btch_size,imgs.shape[0]),:,:,:]
        occlu_list[b:min(b+btch_size, imgs.shape[0]), :, :] = get_batch_outputs(this_batch, version=model_ID, input_checkpoint_path='./SHARED_CODE_logdir/mod_'+str(model_ID)+'_hidden_512/SHARED_CODE_hidden_512_model.ckpt', NAME='SHARED_CODE')
        it += 1
    return occlu_list


if __name__ == '__main__':

    print('OCCLUSION ...')

    occlu_path = '.\SHARED_CODE_occlusion_maps\\'
    if not os.path.exists(occlu_path):
        os.makedirs(occlu_path, exist_ok=True)

    # Set dataset configurations
    test_types = ['00ver', '01square', '07squares', '07square&stars', '01circle', '07circles', '07circle&stars', '01hexagon', '07hexagons', '07hexagon&stars', '21squares_stars_uncrowd', '21squares_stars_crowd']
    shapeMatrices = [[], [1], [1, 1, 1, 1, 1, 1, 1], [6, 1, 6, 1, 6, 1, 6], [2], [2, 2, 2, 2, 2, 2, 2], [6, 2, 6, 2, 6, 2, 6], [3], [3, 3, 3, 3, 3, 3, 3], [6, 3, 6, 3, 6, 3, 6], [[6, 1, 6, 1, 6, 1, 6], [6, 1, 6, 1, 6, 1, 6], [6, 1, 6, 1, 6, 1, 6]], [[1, 6, 1, 6, 1, 6, 1], [6, 1, 6, 1, 6, 1, 6], [1, 6, 1, 6, 1, 6, 1]]]

    for model_ID in range(5):
        for i, shapeMatrix in enumerate(shapeMatrices):
            for offset_type in [0, 1]:  # left vernier, right vernier
                print('\n\nMODEL %1i processing stimulus %s with offset %1i' % (model_ID, test_types[i], offset_type))
                occlusion(shapeMatrix, test_types[i], offset_type, dilation=1, model_ID=model_ID)

    print('We are done.')