# Crowding Reveals Fundamental Differences in Local vs. Global Processing in Humans & Machines

Doerig, Bornet, Choung & Herzog (2019)



# Code to reproduce results


# AlexNet

Run alexnet_with_decoders.py first, to train the networks.

Run make plots to reproduce the plots in figure 3.

Run occlusion_run_alexnet.py to create the occlusion maps .npy files.

Run occlusion_plot_alexnet.py to create the occlusion results images in figure 4 


# ResNet 50: vanila & Geirhos et al.'s shape biased version

Run resnet50_train.py to train the vanila & shape biased ResNet50 networks.

Run resnet50_test.py to test the trained ResNet50 backboned models and reproduce the plots in Figure 3.

Run occlusion_run_resnet50.py to to create the occlusion maps .npy files.

Run occlusion_plot_resnet50.py to create the occlusion results images in figure 4 

