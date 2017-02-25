
layers = [  [3, 3, 3, 16],
            [3, 3, 16, 32],
            [3, 3, 32, 64],
            [3, 3, 64, 128],
            [3, 3, 128, 256],
            [3, 3, 256, 512],
            [3, 3, 512, 1024],
            [3, 3, 1024, 1024],
            [1, 1, 1024, 425]]

count = 0
for layer in layers:
    count += reduce(lambda x, y: x * y, layer)
    count += layer[-1] * 4
count -= 425 * 3
print count

