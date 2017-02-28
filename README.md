# Tiny-YOLO
TensorFlow Version 1.0.0
Reference implementation from [Darknet YOLO](https://pjreddie.com/darknet/yolo/).

# Architecture
Outputs below are printed when using Darknet.
Nonlinearity - Leaky ReLU
```
0: Convolutional Layer: 416 x 416 x 3 image, 16 filters -> 416 x 416 x 16 image
1: Maxpool Layer: 416 x 416 x 16 image, 2 size, 2 stride
2: Convolutional Layer: 208 x 208 x 16 image, 32 filters -> 208 x 208 x 32 image
3: Maxpool Layer: 208 x 208 x 32 image, 2 size, 2 stride
4: Convolutional Layer: 104 x 104 x 32 image, 64 filters -> 104 x 104 x 64 image
5: Maxpool Layer: 104 x 104 x 64 image, 2 size, 2 stride
6: Convolutional Layer: 52 x 52 x 64 image, 128 filters -> 52 x 52 x 128 image
7: Maxpool Layer: 52 x 52 x 128 image, 2 size, 2 stride
8: Convolutional Layer: 26 x 26 x 128 image, 256 filters -> 26 x 26 x 256 image
9: Maxpool Layer: 26 x 26 x 256 image, 2 size, 2 stride
10: Convolutional Layer: 13 x 13 x 256 image, 512 filters -> 13 x 13 x 512 image
11: Maxpool Layer: 13 x 13 x 512 image, 2 size, 1 stride
12: Convolutional Layer: 13 x 13 x 512 image, 1024 filters -> 13 x 13 x 1024 image
13: Convolutional Layer: 13 x 13 x 1024 image, 1024 filters -> 13 x 13 x 1024 image
14: Convolutional Layer: 13 x 13 x 1024 image, 425 filters -> 13 x 13 x 425 image
```

# Usage
Ensure Tiny-YOLO is the base directory everything is run from.
All FLAGS can be parsed in as command lines arguments. For example:

```
python tiny-yolo.py eval_one_image --image_path [/path/to/image] --prob_thresh 0.30
```

Looking inside tiny-yolo.py should be sufficient for most purposes.

# Basic Profiles
All single image processing. No batching involved. 

```
##################
=== TensorFlow ===
##################
GTX 1060 6GB
1: 1.99424117824 FPS
10: 16.3980344115 FPS
100: 61.4526265944 FPS
1000: 82.4555698622 FPS
10000: 87.7515871075 FPS

CPU
1: 8.84767865468 FPS
100: 9.94553707402 FPS

CPU (v1.0.0 with optimizers)
1: 13.788162895 FPS
100: 18.042067404 FPS

CPU (v1.0.0 with optimizers and quantized)
100: 6.43561141096 FPS

##################
==== Darknet =====
##################
NOTE These tests might be trivialized. Simply ran a for loop X times in
line 485 in detector.c.

GTX 1060 6GB
1: 138 FPS
10: 143 FPS
100: 150 FPS
1000: 153 FPS

CPU
1: 0.472595376 FPS
10: 0.047612055 FPS
100: 0.471698113 FPS
```
