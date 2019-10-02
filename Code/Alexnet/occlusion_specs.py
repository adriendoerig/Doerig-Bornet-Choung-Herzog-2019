import matplotlib.pyplot as plt
import random
import numpy as np
from batch_maker import StimMaker
from occlusion_run_alexnet import fixed_noise

def make_occlusion_dataset(shapeMatrix, ratios, slide_range=None, frame_work='tensorflow', dilation=4, offset=None):

    from math import ceil

    imgSize = (227, 227)
    shapeSize = 18

    if shapeMatrix is None or len(shapeMatrix)==0:
        shape_x = 1
        shape_y = 1
    else:
        shape_x         = (np.array(shapeMatrix).shape[1] if len(np.array(shapeMatrix).shape)>1 else np.array(shapeMatrix).shape[0])
        shape_y         = (np.array(shapeMatrix).shape[0] if isinstance(shapeMatrix[0], list) else 1)

    shape_position = (imgSize[0]//2-shapeSize//2,imgSize[0]//2-shapeSize//2)

    # Set sliderange -> overall or range
    if slide_range is not None:
        slide_range = (ceil(shape_position[0] + shapeSize * shape_y * 1.2), ceil(shape_position[1] + shapeSize * shape_x * 1.2))
        print(slide_range)

    batch_images, batch_labels, batch_grids = slide_noise_mask(shapeMatrix, ratios, imgSize=imgSize,
                                                 gapSize=shapeSize, shapeSize=shapeSize, slide_range=slide_range,
                                                 noiseLevel=0.1, shape_position=shape_position, offset=offset, dilation=dilation)

    plot_stimuli = 0
    if plot_stimuli:
        fig, ax = plt.subplots(3)
        for i in range(3):
            ax[i].imshow(batch_images[i,:,:,0])
        plt.show()

    if frame_work is 'pytorch':
        batch_images = np.transpose(batch_images, (0,3,1,2))
    elif frame_work is 'tensorflow':
        batch_images = batch_images

    return batch_images, batch_labels, batch_grids


def slide_noise_mask(shapeMatrix,ratios,imgSize=(227,227), slide_range=None, gapSize=18, shapeSize=18, noiseLevel=0.1, shape_position=(10,10), offset=None, dilation=1):
    '''
    Create images with sliding noise-patch

    Parameters:
        shapeMatrix: configuration matrix
        gapSize: gap for the sliding mask (default to be as same as shape size)
        shapeSize: each shape's size
        noiseLevel: background and noise-patch noise level
        offset: vernier offset (0: left, 1: right)
        dilation: sliding dilation (default to be pass-through each pixel)

    noise patch size: (shapeSize//2 x shapeSize//2) by default, and has same noise level of background
    noise_patch: stores h x w starting positions

    returns: np.array of slided images and label of this set of image
    '''
    if offset is None:
        offset = random.randint(0,1) # using fixed offset

    barWidth = 1
    rufus = StimMaker(imgSize, shapeSize, barWidth)

    noise_patch_size = shapeSize//3
    noise_patch = np.random.normal(0, .1, size=(noise_patch_size, noise_patch_size))

    noise_imgs, noise_label = rufus.generate_Batch(1, ratios, noiseLevel=noiseLevel, shapeMatrix=shapeMatrix, fixed_position=shape_position, offset=offset, offset_size=2, fixed_noise=fixed_noise)

    if slide_range is None:  # slide noise all over locations
        slide_range = imgSize
        counter = 0
        n_imgs_total = int(len(range(0, slide_range[0] - shapeSize - 1, dilation))*len(range(0, slide_range[1] - shapeSize - 1, dilation)))  # -1 is needed for size adjustments. depending on dilation and patch_size, you may need to remove it
        noise_imgs = np.tile(noise_imgs, [n_imgs_total+1, 1, 1, 1])  # +1 because the 0th image is the original one, without noise
        for r in range(0, slide_range[0] - shapeSize - 1, dilation):  # -1 is needed for size adjustments. depending on dilation and patch_size, you may need to remove it
            for c in range(0, slide_range[1] - shapeSize - 1, dilation):  # -1 is needed for size adjustments. depending on dilation and patch_size, you may need to remove it
                noise_imgs[counter+1, r:r+noise_patch_size, c:c+noise_patch_size, :] = np.tile(np.expand_dims(noise_patch, axis=-1), [1,1,1,3])
                counter += 1
                if counter % 1000 == 0:
                    print("\rProgress: {:.1f} ...%".format(counter * 100 / n_imgs_total, end=" "))
        noise_grid = [np.arange(0, slide_range[0] - shapeSize - 1, dilation), np.arange(0, slide_range[1] - shapeSize - 1, dilation)]  # -1 is needed for size adjustments. depending on dilation and patch_size, you may need to remove it

        # save a video of your stimulus as .gif to check it everything's fine.
        save_gif = 0
        if save_gif:
            import imageio
            gif_imgs = noise_imgs[::10,:,:,:].copy()
            gif_imgs = 255*(gif_imgs-gif_imgs.min())/(gif_imgs.max()-gif_imgs.min())  # normalize for uint8 conversion (needed for gif)
            imageio.mimsave('./stimuli_stim_' + str(shapeMatrix) + '.gif', gif_imgs.astype(np.uint8), fps=5)

    return noise_imgs, noise_label, noise_grid
