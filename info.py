# Text from darknet
# Nonlinearity that they use is leaky relu
# 0: Convolutional Layer: 416 x 416 x 3 image, 16 filters -> 416 x 416 x 16 image
# 1: Maxpool Layer: 416 x 416 x 16 image, 2 size, 2 stride
# 2: Convolutional Layer: 208 x 208 x 16 image, 32 filters -> 208 x 208 x 32 image
# 3: Maxpool Layer: 208 x 208 x 32 image, 2 size, 2 stride
# 4: Convolutional Layer: 104 x 104 x 32 image, 64 filters -> 104 x 104 x 64 image
# 5: Maxpool Layer: 104 x 104 x 64 image, 2 size, 2 stride
# 6: Convolutional Layer: 52 x 52 x 64 image, 128 filters -> 52 x 52 x 128 image
# 7: Maxpool Layer: 52 x 52 x 128 image, 2 size, 2 stride
# 8: Convolutional Layer: 26 x 26 x 128 image, 256 filters -> 26 x 26 x 256 image
# 9: Maxpool Layer: 26 x 26 x 256 image, 2 size, 2 stride
# 10: Convolutional Layer: 13 x 13 x 256 image, 512 filters -> 13 x 13 x 512 image
# 11: Maxpool Layer: 13 x 13 x 512 image, 2 size, 1 stride
# 12: Convolutional Layer: 13 x 13 x 512 image, 1024 filters -> 13 x 13 x 1024 image
# 13: Convolutional Layer: 13 x 13 x 1024 image, 1024 filters -> 13 x 13 x 1024 image
# 14: Convolutional Layer: 13 x 13 x 1024 image, 425 filters -> 13 x 13 x 425 image
# 15: side: Using default '13'
# side: Using default '13'
# Region Layer
# Unused field: 'absolute = 1'
# Unused field: 'random = 1'
# Loading weights from tiny-yolo.weights...Done!
