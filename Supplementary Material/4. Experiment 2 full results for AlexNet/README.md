# 4. Experiment 2 full results for AlexNet


In experiment 2, we measured where is the core area to classify vernier offsets from activity in AlexNet (Krizhevsky et al. 2012) layers, by using occluded stimuli. We tested additional stimuli with 1 circle, 7 circles, 1 hexagon, 7 hexagons, 7 alternating circles and stars, and 7 alternating hexagons and stars. Below are the occlusion sensitivity measured and averaged over all layers (the same measure as in Figure 4). The result shows a consistent pattern with figure 4, where indicating such a local computation of AlexNet is not restricted to stimulus type, but general across various flankers (shapes).


The _combined files contain the maps computed as described in the paper's methods, for each layer, averaged over 5 trained networks. Values are normalized in [0,1] for each layer.

The _threshold files contain the map averaged across layers and 5 trained networks, with a threshold applied at 0.4*[max map value]. Values are normalized in [0,1].

The .gif files contain the same map as the _threshold image, but the threshold is varies continuously, to show how the choice of threshold value affects results. Values are normalized in [0,1].


Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105). 