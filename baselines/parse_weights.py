import numpy as np

f = open('/home/jkschin/code/github-others/darknet/tiny-yolo.weights', 'rb')
for i in xrange(4):
    print np.frombuffer(f.read(4), np.int32)

arr = np.fromfile(f, np.float32)
print arr.shape
# for i in xrange(len(arr)):
#     print arr[i]
# arr = np.fromfile(f, np.float32)
# for i in xrange(100):
#     print arr[i]
