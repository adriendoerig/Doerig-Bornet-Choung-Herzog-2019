# 5. Experiment 2 full results for ResNet50


In experiment 2, we measured where is the core area for a decoder to do the task correctly, using occluded stimuli. Here are additional results for the 10 layers of ResNet50 on which we performed our analysis (the conv3 layer in the first bottleneck, and after the 9 first bottlenecks, see He et al. 2016). Using more layers (i.e., decoding after all bottlenecks instead of only the first 9) introduces a lot of noise, but yields the same qualitative result: the netowrks' decisions are based on the local target neighborhood. We tested additional stimuli with 1 circle, 7 circles, 1 hexagon, 7 hexagons, 7 alternating circles and stars, 7 alternating hexagons and stars and two 3x7 configurations of squares and stars (one leading to crowding and the other to uncrowding in humans). Below are the occlusion sensitivity measured and averaged over all layers (the same measure as in Figure 4). The result shows a consistent pattern with figure 4, indicating that our results are not restricted to stimulus type, but general across various flankers (shapes).

The _18layers files show results averaged over 18 layers. The other files show results averaged over ten layers as described above.

The _combined files contain the maps computed as described in the paper's methods, for each layer, averaged over 5 trained networks. Values are normalized in [0,1] for each layer.
The _threshold files contain the map averaged across layers and 5 trained networks, with a threshold applied at 0.4*[max map value]. Values are normalized in [0,1].
The .gif files contain the same map as the _threshold image, but the threshold is varies continuously, to show how the choice of threshold value affects results. Values are normalized in [0,1].

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).