# 3. Experiment 1 results for Geirhos et al.'s shape-biased ResNet50


Gheiros et al. trained ResNet50 with stylized-ImageNet (SIN) dataset. SIN was made to let network to focus only on the shape, but texture. Therefore, Gheiros and colleagues switched the texture of each shape in ImageNet (IN), to allow each shape to have numerous textures, by using AdaIN style transfer (Huang & Delongie, 2017). Among the Gheiros et al.’s shape-based pre-trained networks, we used the model_C ("resnet50 trained on SIN and IN then finetuned on IN"), which was the most robust and accurate to the noises. 
We reproduced experiment 1 in this network, we decoded vernier offset direction from 18 layers of ResNet50 (an early after the third convolution in the first "bottleneck", and after each "bottleneck" unit, see He et al. 2016) for a variety of flanker types. Below are the results for decoding performance on each layer. The following plots are presented as in figure 3. The results show similar tendency with figure 3.

Files show the same plot as panel c in figure 3

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
Huang, X., & Belongie, S. (2017). Arbitrary style transfer in real-time with adaptive instance normalization. In Proceedings of the IEEE International Conference on Computer Vision (pp. 1501-1510).
Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A., & Brendel, W. (2018). ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. arXiv preprint arXiv:1811.12231.

