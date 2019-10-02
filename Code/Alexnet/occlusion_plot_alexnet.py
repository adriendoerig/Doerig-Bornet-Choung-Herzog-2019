import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
import os
from occlusion_specs import make_occlusion_dataset

save_combined_maps = 1
save_thresholded_gifs = 1
save_thresholded_maps = 1

stim_names = ['00ver', '01square', '07squares', '07square&stars', '01circle', '07circles', '07circle&stars', '01hexagon', '07hexagons', '07hexagon&stars', '21squares_stars_uncrowd', '21squares_stars_crowd']
stim_matrices = [[], [1], [1, 1, 1, 1, 1, 1, 1], [6, 1, 6, 1, 6, 1, 6], [2], [2, 2, 2, 2, 2, 2, 2], [6, 2, 6, 2, 6, 2, 6], [3], [3, 3, 3, 3, 3, 3, 3], [6, 3, 6, 3, 6, 3, 6], [[6, 1, 6, 1, 6, 1, 6], [6, 1, 6, 1, 6, 1, 6], [6, 1, 6, 1, 6, 1, 6]], [[1, 6, 1, 6, 1, 6, 1], [6, 1, 6, 1, 6, 1, 6], [1, 6, 1, 6, 1, 6, 1]]]
occlumap_dir = './SHARED_CODE_occlusion_maps/'
occlumap_files = os.listdir(occlumap_dir)
n_classifiers = 9
dilation = 1  # used for naming the plot adequately

for i, stim_name in enumerate(stim_names):

    model_combined_maps = []
    map_name = os.path.join(occlumap_dir, '___occlu_' + str(stim_name) + '_' + str(dilation) + '_combined')
    stim_img, _, _ = make_occlusion_dataset(stim_matrices[i], ratios=[0,0,0,1], dilation=200, offset=0)  # stim_imgs[0] contains the exact stimulus used to create the occlusion maps

    for model_ID in range(5):
        model_occlumaps = []
        these_files = [f for f in occlumap_files if (stim_name in f and f[-5] == str(model_ID) and 'combined' not in f and 'npy' in f)]
        print('Current files: ' + str(these_files))

        for j, offset_type in enumerate(['0', '1']):

            try:
                this_map = np.load(occlumap_dir+these_files[j])
            except:
                print(occlumap_dir+these_files[j] + ' not found.')
                raise(NameError)

            model_occlumaps.append(this_map)

        model_combined_maps.append(model_occlumaps[0]/2+model_occlumaps[1]/2)

    final_combined_map = np.mean(np.array(model_combined_maps), axis=0)

    max_score = 1
    patch_size = 6
    padding = patch_size // 2
    stim_img = stim_img[0, padding:-padding, padding:-padding, 0]

    # plot and save figures + occlumap
    if save_combined_maps:
        plt.figure(figsize=(20, 20))
        for c in range(n_classifiers):
            plt.subplot(3, 3, c+1)
            occlu_img = np.zeros_like(stim_img)
            occlu_img[:final_combined_map[c, :, :].shape[0], :final_combined_map[c, :, :].shape[1]] = final_combined_map[c, :, :]
            plt.imshow(30 * occlu_img, cmap='RdBu', norm=mplt.colors.Normalize(vmin=-max_score, vmax=max_score))
            plt.colorbar()
            plt.imshow(stim_img, cmap='Greys', norm=mplt.colors.Normalize(vmin=0, vmax=max_score), alpha=0.1)
            plt.axis('off')
            plt.title('Layer' + str(c + 1))
        plt.savefig(map_name + '.png', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

    if save_thresholded_gifs:
        # plot using sum of all classifiers
        import imageio
        threshs  = np.arange(0.1, 1.0, 0.01)
        gif_imgs = []
        fig = plt.figure(figsize=(20, 20))
        for idx, thresh in enumerate(threshs):

            occlu_img = np.zeros_like(stim_img)
            occlu_img[:final_combined_map[0, :, :].shape[0], :final_combined_map[0, :, :].shape[1]] = np.sum(final_combined_map, axis=0)
            threshold = thresh * np.abs(occlu_img).max()
            occlu_img[np.abs(occlu_img) < threshold] = 0.0

            plt.imshow(30 * occlu_img, cmap='RdBu', norm=mplt.colors.Normalize(vmin=-max_score, vmax=max_score))
            plt.colorbar()
            plt.imshow(stim_img, cmap='Greys', norm=mplt.colors.Normalize(vmin=0, vmax=max_score), alpha=0.1)
            plt.axis('off')
            plt.title('Sum of decoders over layers, thresh is %.2f of max abs value.' % (thresh,))
            fig.canvas.draw()  # draw the canvas, cache the renderer

            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            gif_imgs.append(image)
            fig.clear()

        imageio.mimsave(map_name + '___sum_of_decoders_local_norm.gif', gif_imgs, fps=24)

    if save_thresholded_maps:
        plt.figure(figsize=(20, 20))
        occlu_img = np.zeros_like(stim_img)
        occlu_img[:final_combined_map[0, :, :].shape[0], :final_combined_map[0, :, :].shape[1]] = np.sum(final_combined_map, axis=0)
        thresh = .4
        threshold = thresh*np.abs(occlu_img).max()  # thresholding relative to the largest value in occlu_img
        occlu_img[np.abs(occlu_img) < threshold] = 0.0
        plt.imshow(30 * occlu_img, cmap='RdBu', norm=mplt.colors.Normalize(vmin=-max_score, vmax=max_score))
        plt.colorbar()
        plt.imshow(stim_img, cmap='Greys', norm=mplt.colors.Normalize(vmin=0, vmax=max_score), alpha=0.1)
        plt.axis('off')
        plt.title('Sum of decoders over layers, thresh is %.2f of max abs value.' % (thresh,))
        plt.savefig(map_name + '___sum_of_decoders___threshold.png', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()

print('Finished creating & saving combined occlusion maps.')
