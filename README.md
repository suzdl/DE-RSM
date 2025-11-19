# DE-RSM
A depth map enhancement model. We train it within the relatively large-scale data supervision of low-quality depth GT and the color-invariant structure supervision of RGB, in order to not overfit the low-quality depth GT.

![image](https://github.com/dangdang17/DE-RSM/assets/78062148/da8c9d28-7367-48f2-904f-81159ac3ecfa)

Fig. 1. An example of depth GT in NYUv2. (a) RGB, (b) GT, (c) enhanced GT by our model. (d) edges of (a) and (b). (e) edges of (a) and (c). The edges of depth GT in red and RGB in white are misaligned in (d) while they are well consistent in (e).

![color_invariant](https://github.com/suzdl/DE-RSM/assets/78062148/3cb8cb54-af71-4567-ab9f-8a55fd6863b5)

Fig. 2. Explanation of color-invariant. The real-world scenes have the characteristics of multi-RGB-with-one-depth, which is ill-posed. We further develop a structure model. It not only does well in dealing the ill-posed relationship on RGB-D, but also helps not overfitting the low-quality depth GT. 

## Run
Download the [pretrained weights](https://drive.google.com/drive/folders/1o3vboZ20PhnOxLFP7U6trWJEdrJG8n3d?hl=zh_CN) inside the folder 'models'.

The requirements of the environment are Python==3.8, Pytorch==2.0.

Run the following code for depth map enhancement.

```python test_enhance_realscenes.py```

More test sets and evaluation code can be found [here](https://github.com/Wang-xjtu/G2-MonoDepth).

The training code is an example of train_full_aim16.py.
