import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from matplotlib import pyplot as plt

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

# weights_folder = '/home/jkschin/code/Tiny-YOLO/tiny-yolo-weights/*.csv'
# for weights_file in sorted(glob(weights_folder)):
#     weights_trained = np.genfromtxt(weights_file, delimiter=',', dtype=np.float32)
#     print weights_file, weights_trained.shape

load_csv = True
save = False
debug = True
def leaky_relu(x, scalar):
    return tf.maximum(x, tf.scalar_mul(scalar, x))

def get_name(layer_name, counters):
    if layer_name not in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

@add_arg_scope
def conv_and_bias(n, inp, filter_size, num_filters, stride, padding, nonlinearity, batch_norm, counters):
    inp_shape = map(lambda x: int(x), inp.get_shape())
    conv_shape = [filter_size, filter_size, inp_shape[-1], num_filters]
    bias_shape = [num_filters]
    mean_shape = [num_filters]
    variance_shape = [num_filters]
    scales_shape = [num_filters]

    conv_weights = tf.get_variable(get_name('conv', counters), conv_shape)
    #TODO remove the constant initializer after debug
    bias_weights = tf.get_variable(get_name('bias', counters), bias_shape, initializer=tf.constant_initializer(0))

    if batch_norm:
        scale_weights = tf.get_variable(get_name('scale', counters), scales_shape)
        mean_weights = tf.get_variable(get_name('mean', counters), mean_shape)
        variance_weights = tf.get_variable(get_name('variance', counters), variance_shape)

    # if load_csv:
    #     args = (conv_weights, bias_weights, mean_weights, variance_weights, scales_weights)
    #     conv_weights, bias_weights = load_csv_weights(n, conv_shape, bias_shape, conv_weights, bias_weights)
        # load_csv_weights(n, conv_shape, bias_shape, conv_weights, bias_weights)

    #TODO Darknet has an explicit padding method that we might want to mimic.
    #TODO tried this but has no effect on the weights
    # d0_pad = int(conv_shape[0]/2)
    # d1_pad = int(conv_shape[1]/2)
    # inp_padded = tf.pad(inp, paddings=[[0,0], [d0_pad, d0_pad], [d1_pad, d1_pad], [0,0]])
    conv = tf.nn.conv2d(inp, conv_weights, strides=[1, stride, stride, 1], padding='SAME')
    if batch_norm:
        conv = tf.nn.batch_normalization(conv, mean_weights, variance_weights, offset=0, scale=scale_weights, variance_epsilon=0.00001)
    conv_bias = tf.add(conv, bias_weights)
    return conv_bias if nonlinearity is None else nonlinearity(conv_bias, 0.1)

# def load_csv_weights(n, conv_shape, bias_shape, conv_weights_vars, bias_weights_vars):
#     conv_weights = np.empty(conv_shape, dtype=np.float32)
#     conv_file = '/home/jkschin/code/Tiny-YOLO/tiny-yolo-weights/conv_weight_layer%d.csv' %n
#     print 'Loading: ', conv_file
#     conv_trained = np.genfromtxt(conv_file, delimiter=',', dtype=np.float32)
#     for i in range(conv_shape[0]):
#         for j in range(conv_shape[1]):
#             for k in range(conv_shape[2]):
#                 for l in range(conv_shape[3]):
#                     conv_weights[i, j, k, l] = conv_trained[(l * conv_shape[0] * conv_shape[1] * conv_shape[2]) + (k * conv_shape[0] * conv_shape[1]) + (i * conv_shape[0]) + j]

#     bias_weights = np.empty(bias_shape, dtype=np.float32)
#     bias_file = '/home/jkschin/code/Tiny-YOLO/tiny-yolo-weights/conv_bias_layer%d.csv' %n
#     print 'Loading: ', bias_file
#     bias_trained = np.genfromtxt(bias_file, delimiter=',', dtype=np.float32)
#     for i in range(bias_shape[0]):
#         bias_weights[i] = bias_trained[i]
#     conv_weights_vars = tf.assign(conv_weights_vars, conv_weights)
#     bias_weights_vars = tf.assign(bias_weights_vars, bias_weights)
#     return conv_weights_vars, bias_weights_vars

def load_4d(sess, shape, conv_weights_vars, pretrained_weights):
    conv_weights = np.empty(shape, dtype=np.float32)
    # YoloTensorflow229 implementation
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    conv_weights[i, j, k, l] = pretrained_weights[(l * shape[0] * shape[1] * shape[2]) + (k * shape[0] * shape[1]) + (i * shape[0]) + j]
    # For this implementation, if the shape is [4, 4, 3, 16]:
    # Filter 1
    # [0, 1, 2, 3, ... 14, 15]
    # [256, 257, ... 270, 271]
    # [512, 513, ... 526, 527]
    #
    # Filter 2
    # [16, 17, ... 30, 31]
    # [272, 273 ... 286, 287]
    # [528, 529 ... 542, 543]
    # The activations were huge, probably wrong.
    # for i in range(shape[2]):
    #     for j in range(shape[3]):
    #         for k in range(shape[0]):
    #             for l in range(shape[1]):
    #                 conv_weights[k,l,i,j] = pretrained_weights[(k*shape[1] + l + i*shape[0]*shape[1]*shape[3] + j*shape[0]*shape[1])]
    return sess.run(tf.assign(conv_weights_vars, conv_weights))

def load_1d(sess, shape, weights_vars, pretrained_weights):
    weights = np.empty(shape, dtype=np.float32)
    for i in range(shape[0]):
        weights[i] = pretrained_weights[i]
        # print pretrained_weights[i]
    return sess.run(tf.assign(weights_vars, weights))

def load_from_binary(sess):
    f = open('/home/jkschin/code/github-others/darknet/tiny-yolo.weights', 'rb')
    print "Reading junk out..."
    for i in xrange(4):
        print np.frombuffer(f.read(4), np.int32)
    idx = 0
    pretrained_weights = np.fromfile(f, np.float32)
    var_list = ['bias', 'scale', 'mean', 'variance', 'conv']
    with tf.variable_scope('model', reuse=True):
        for i in xrange(8):
            for var_name in var_list:
                if var_name == 'conv':
                    var_name = var_name + '_' + str(i)
                    var = tf.get_variable(var_name)
                    shape = map(lambda x: int(x), var.get_shape())
                    end_idx = reduce(lambda x, y: x * y, shape)
                    var = load_4d(sess, shape, var, pretrained_weights[idx:idx+end_idx])
                    print "Loading %s from %d to %d" %(var_name, idx, idx+end_idx)
                    print shape
                    idx += end_idx
                else:
                    var_name = var_name + '_' + str(i)
                    var = tf.get_variable(var_name)
                    shape = map(lambda x: int(x), var.get_shape())
                    end_idx = reduce(lambda x, y: x * y, shape)
                    var = load_1d(sess, shape, var, pretrained_weights[idx:idx+end_idx])
                    print "Loading %s from %d to %d" %(var_name, idx, idx+end_idx)
                    idx += end_idx

        for i in xrange(8, 9):
            for var_name in var_list:
                if var_name == 'bias':
                    var_name = var_name + '_' + str(i)
                    var = tf.get_variable(var_name)
                    shape = map(lambda x: int(x), var.get_shape())
                    end_idx = reduce(lambda x, y: x * y, shape)
                    var = load_1d(sess, shape, var, pretrained_weights[idx:idx+end_idx])
                    print "Loading %s from %d to %d" %(var_name, idx, idx+end_idx)
                    idx += end_idx
                elif var_name == 'conv':
                    var_name = var_name + '_' + str(i)
                    var = tf.get_variable(var_name)
                    shape = map(lambda x: int(x), var.get_shape())
                    end_idx = reduce(lambda x, y: x * y, shape)
                    var = load_4d(sess, shape, var, pretrained_weights[idx:idx+end_idx])
                    print "Loading %s from %d to %d" %(var_name, idx, idx+end_idx)
                    idx += end_idx
    assert idx == 16175385

    # nodes = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # for node in nodes:
    #     print node.name

def print_conv_weights(sess):
    nodes = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for node in nodes:
        print node.name
        node_val = sess.run(node)
        print node_val

def model_spec(image):
    layers=[image]
    counters = {}
    with arg_scope([conv_and_bias], batch_norm=True, counters=counters):
        layers.append(conv_and_bias(0, layers[-1], 3, 16, 1, 'SAME', leaky_relu))
        layers.append(tf.nn.max_pool(layers[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'VALID'))
        layers.append(conv_and_bias(2, layers[-1], 3, 32, 1, 'SAME', leaky_relu))
        layers.append(tf.nn.max_pool(layers[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'VALID'))
        layers.append(conv_and_bias(4, layers[-1], 3, 64, 1, 'SAME', leaky_relu))
        layers.append(tf.nn.max_pool(layers[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'VALID'))
        layers.append(conv_and_bias(6, layers[-1], 3, 128, 1, 'SAME', leaky_relu))
        layers.append(tf.nn.max_pool(layers[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'VALID'))
        layers.append(conv_and_bias(8, layers[-1], 3, 256, 1, 'SAME', leaky_relu))
        layers.append(tf.nn.max_pool(layers[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'VALID'))
        layers.append(conv_and_bias(10, layers[-1], 3, 512, 1, 'SAME', leaky_relu))
        layers.append(tf.nn.max_pool(layers[-1], [1, 2, 2, 1], [1, 1, 1, 1], 'SAME'))
        layers.append(conv_and_bias(12, layers[-1], 3, 1024, 1, 'SAME', leaky_relu))
        layers.append(conv_and_bias(13, layers[-1], 3, 1024, 1, 'SAME', leaky_relu))
        layers.append(conv_and_bias(14, layers[-1], 1, 425, 1, 'SAME', None, batch_norm=False))
    logits = layers[-1]
    merge = []
    for i in xrange(5):
        base = i*85
        merge.append(logits[:, :, :, base:base+4])
        merge.append(tf.expand_dims(tf.sigmoid(logits[:, :, :, base+4]), -1))
        merge.append(tf.nn.softmax(logits[:, :, :, base+5:base+85]))
    return tf.concat(3, merge)

def predictions(logits, class_feed):
    num_classes = 80
    num_bboxes = 5
    anchor_priors = np.array([[0.738768, 0.874946], [2.42204, 2.65704], [4.30971, 7.04493], [
10.246, 4.59428],  [12.6868, 11.8741]])
    coords = 4
    prob_thresh = 0.8
    iou_thresh = 0.8
    # TODO hardcoded right now
    w = 1920
    h = 1080

    # bboxes_logits has shape of [B, N, H, W, 4]
    bboxes_logits = tf.stack([logits[:, :, :, i*num_classes + 0:(i+0)*num_classes + 4] for i in xrange(num_bboxes)], 0)
    bboxes_shape = map(lambda f: int(f), bboxes_logits.get_shape())

    # Assume:
    #     bboxes_shape[2] = 5
    #     bboxes_shape[3] = 7
    # row_grid_coords gives this:
    # array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
    #        [ 2.,  2.,  2.,  2.,  2.,  2.,  2.],
    #        [ 3.,  3.,  3.,  3.,  3.,  3.,  3.],
    #        [ 4.,  4.,  4.,  4.,  4.,  4.,  4.]])

    # col_grid_coords gives this:
    # array([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
    #        [ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
    #        [ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
    #        [ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
    #        [ 0.,  1.,  2.,  3.,  4.,  5.,  6.]])
    row_grid_coords = tf.constant(np.reshape(np.arange(0, bboxes_shape[2]), (bboxes_shape[2], 1)) + np.zeros((bboxes_shape[2], bboxes_shape[3])), dtype=tf.float32)
    col_grid_coords = tf.constant(np.reshape(np.arange(0, bboxes_shape[3]), (1, bboxes_shape[3])) + np.zeros((bboxes_shape[2], bboxes_shape[3])), dtype=tf.float32)

    # removed the divide by w and divide by h for now. not sure why darknet places it in such a funny way.
    # one reason might be because of the way they plot bounding boxes. at least for TF, tf.image.draw_bounding_boxes takes in boxes at floating points
    x1s = (tf.sigmoid(bboxes_logits[:, :, :, :, 0]) + col_grid_coords) / bboxes_shape[3]
    y1s = (tf.sigmoid(bboxes_logits[:, :, :, :, 1]) + row_grid_coords) / bboxes_shape[2]
    x2s = x1s + (tf.exp(bboxes_logits[:, :, :, :, 2]) * reduce(lambda x, _: np.expand_dims(x, -1), xrange(3), anchor_priors[:,0]) / bboxes_shape[3])
    y2s = y1s + (tf.exp(bboxes_logits[:, :, :, :, 3]) * reduce(lambda x, _: np.expand_dims(x, -1), xrange(3), anchor_priors[:,1]) / bboxes_shape[2])
    # note that tf.image.non_max_suppression takes in the coordinates in this format.
    bboxes = tf.stack([y1s, x1s, y2s, x2s], axis=-1)

    # scale_logits has shape of [B, N, H, W, 1]
    scale_logits = tf.stack([logits[:, :, :, i*num_classes + 4:(i+0)*num_classes + 5] for i in xrange(num_bboxes)], 0)
    # scale_logits = tf.sigmoid(scale_logits)
    # with tf.control_dependencies([tf.assert_less_equal(scale_logits, 1.0), tf.assert_greater_equal(scale_logits, 0.0)]):
        # scale_logits = tf.identity(scale_logits)

    tf.add_to_collection('scale_logits', scale_logits)

    # cls_logits has shape of [B, N, H, W, 80]
    cls_logits = tf.stack([logits[:, :, :, i*num_classes + 5:(i+1)*num_classes + 5] for i in xrange(num_bboxes)], 0)
    # cls_logits = tf.nn.softmax(cls_logits, -1)
    # with tf.control_dependencies([tf.assert_less_equal(cls_logits, 1.0), tf.assert_greater_equal(cls_logits, 0.0)]):
    #     cls_logits = tf.identity(cls_logits)

    # # TODO check the behaviour of broadcasting. this might be a source of error.
    # # probabilities has shape of [5, N, 13, 13, 80]
    probabilities = cls_logits * scale_logits
    # tf.assert_less_equal(probabilities, 1.0)
    # tf.assert_greater_equal(probabilities, 0.0)

    probs_shape = map(lambda f: int(f), probabilities.get_shape())
    probabilities_bool = tf.greater(probabilities, prob_thresh)
    probabilities = tf.where(probabilities_bool, probabilities, tf.zeros(probs_shape))
    tf.add_to_collection('probabilities', probabilities)

    # # car is index 2 so let's do that first
    nms_boxes = tf.reshape(bboxes, [reduce(lambda x, y: x*y, bboxes_shape[:-1]), bboxes_shape[-1]])
    nms_probs = tf.reshape(probabilities[:, :, :, :, class_feed], [reduce(lambda x, y: x*y, probs_shape[:-1])])
    bboxes_indices = tf.image.non_max_suppression(nms_boxes, nms_probs, 10, iou_thresh)
    bboxes_output = tf.gather(nms_boxes, bboxes_indices)

    if debug:
        print 'bboxes_logits: ', bboxes_logits.get_shape()
        print 'row_grid_coords: ', row_grid_coords.get_shape()
        print 'col_grid_coords: ', col_grid_coords.get_shape()
        print 'x1s: ', x1s.get_shape()
        print 'y1s: ', y1s.get_shape()
        print 'x2s: ', x2s.get_shape()
        print 'y2s: ', y2s.get_shape()
        print 'bboxes: ', bboxes.get_shape()
        print 'scale_logits: ', scale_logits.get_shape()
        print 'cls_logits: ', cls_logits.get_shape()
        print 'probabilities: ', probabilities.get_shape()
        print 'nms_boxes: ', nms_boxes.get_shape()
        print 'nms_probs: ', nms_probs.get_shape()
        print 'bboxes_indices: ', bboxes_indices.get_shape()
        print 'bboxes_output: ', bboxes_output.get_shape()

    return bboxes_output
    # for i in xrange(num_bboxes):
    #     for j in xrange(bboxes_shape[1]):
    #         for k in range(num_classes):
    #     for j in xrange(num_classes):
    #         nms_boxes = tf.reshape(bboxes_vars[i, :, :, :, 4], [bboxes_shape[0]*bboxes_shape[1]*bboxes_shape[2]*bboxes_shape[3], 4])
    #         nms_probs = tf.reshape(probabilities[i, :, :, :, j], [probs_shape[0]*probs_shape[1]*probs_shape[2]*probs_shape[3]])
    #         nms_outputs = tf.image.non_max_suppression(nms_boxes, nms_probs, 10, iou_thresh)


    # for box_idx in xrange(bboxes_shape[0]): # box index
    #     for i in xrange(bboxes_shape[1]): # batch
    #         for j in xrange(bboxes_shape[2]): # height
    #             for k in xrange(bboxes_shape[3]): # width
    #                 print box_idx, i, j, k
    #                 x1 = (j + tf.sigmoid(bboxes_logits[box_idx, i, j, k, 0])) / w
    #                 y1 = (k + tf.sigmoid(bboxes_logits[box_idx, i, j, k, 1])) / h
    #                 x2 = x1 + tf.exp(bboxes_logits[box_idx, i, j, k, 2]) * anchor_priors[box_idx][0] / w
    #                 y2 = y1 + tf.exp(bboxes_logits[box_idx, i, j, k, 3]) * anchor_priors[box_idx][1] / h
                        # bboxes_vars[box_idx, i, j, k, l+0] = tf.assign(bboxes_vars[box_idx, i, j, k, l+0], x1)
                        # bboxes_vars[box_idx, i, j, k, l+1] = tf.assign(bboxes_vars[box_idx, i, j, k, l+1], y1)
                        # bboxes_vars[box_idx, i, j, k, l+2] = tf.assign(bboxes_vars[box_idx, i, j, k, l+2], x2)
                        # bboxes_vars[box_idx, i, j, k, l+3] = tf.assign(bboxes_vars[box_idx, i, j, k, l+3], y2)

def main(argv):
    image = tf.placeholder(tf.float32, shape=[1, 416, 416, 3])
    class_feed = tf.placeholder(tf.int32, shape=[])
    model = tf.make_template('model', model_spec)
    logits = model(image)
    bboxes = predictions(logits, class_feed)
    image_bbox = tf.image.draw_bounding_boxes(image, tf.expand_dims(bboxes, 0))
    sess = tf.Session()
    saver = tf.train.Saver()
    if load_csv:
        sess.run(tf.global_variables_initializer())
        load_from_binary(sess)
        print_conv_weights(sess)
    else:
        saver.restore(sess, '/home/jkschin/code/Tiny-YOLO/tiny-yolo-model.ckpt')

    # image_in = cv2.imread('/media/jkschin/WD2TB/data/20170206-lornie-road/7290_frames/290_0001.jpg').astype(np.float32)
    image_in = cv2.imread('sample.jpg').astype(np.float32)
    # YOLO original does processing in RGB
    image_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
    image_in = cv2.resize(image_in, (416, 416))
    image_in = np.expand_dims(image_in, 0)
    # darknet scales values from 0 to 1
    image_in = (image_in / 255.0)
    probs = tf.get_collection('probabilities')
    scale_logits = tf.get_collection('scale_logits')
    # scale_logits_val = sess.run(scale_logits, feed_dict={image:image_in})[0]
    # print scale_logits_val
    # print scale_logits_val.shape

    logits_val = sess.run(logits, feed_dict={image:image_in})[0]
    for i in range(13):
        for j in range(13):
            for k in range(425):
                print k+j*425+i*13*425, logits_val[i, j, k]
    # for idx, val in enumerate(logits_val.flatten('F')):
    #     print idx, val
    # print logits_val.shape
    # for class_feed_in in xrange(2, 3):
    #     logits_val, bboxes_val, image_bbox_val, probs_val = sess.run([logits, bboxes, image_bbox, probs], feed_dict={image: image_in, class_feed: class_feed_in})
    #     plt.imshow(np.squeeze(image_bbox_val))
    #     plt.show()
    # if save:
    #     saver.save(sess, '/home/jkschin/code/Tiny-YOLO/tiny-yolo-model.ckpt')
    # print bboxes_val
    # print probs_val[0][:, :, :, :, 2]




if __name__ == '__main__':
    tf.app.run()



