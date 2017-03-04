import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework.python.ops import add_arg_scope

FLAGS = tf.app.flags.FLAGS

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
    bias_weights = tf.get_variable(get_name('bias', counters), bias_shape)

    if batch_norm:
        scale_weights = tf.get_variable(get_name('scale', counters), scales_shape)
        mean_weights = tf.get_variable(get_name('mean', counters), mean_shape)
        variance_weights = tf.get_variable(get_name('variance', counters), variance_shape)

    conv = tf.nn.conv2d(inp, conv_weights, strides=[1, stride, stride, 1], padding=padding)
    if batch_norm:
        conv = tf.nn.batch_normalization(conv, mean_weights, variance_weights, offset=None, scale=scale_weights, variance_epsilon=0.00001)
    conv_bias = tf.add(conv, bias_weights)
    return conv_bias if nonlinearity is None else nonlinearity(conv_bias, 0.1)

def process_logits(logits):
    anchor_priors = np.array([[0.738768, 0.874946], [2.42204, 2.65704], [4.30971, 7.04493], [10.246, 4.59428], [12.6868, 11.8741]])
    # bboxes_logits has shape of [B, N, H, W, 4]
    bboxes_logits = tf.stack([logits[:, :, :, i*(FLAGS.num_classes+5):i*(FLAGS.num_classes+5) + 4] for i in xrange(FLAGS.num_bboxes)], 0)
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

    x1s = (tf.sigmoid(bboxes_logits[:, :, :, :, 0]) + col_grid_coords) / bboxes_shape[3]
    y1s = (tf.sigmoid(bboxes_logits[:, :, :, :, 1]) + row_grid_coords) / bboxes_shape[2]
    ws = (tf.exp(bboxes_logits[:, :, :, :, 2]) * reduce(lambda x, _: np.expand_dims(x, -1), xrange(3), anchor_priors[:,0]) / bboxes_shape[3])
    hs = (tf.exp(bboxes_logits[:, :, :, :, 3]) * reduce(lambda x, _: np.expand_dims(x, -1), xrange(3), anchor_priors[:,1]) / bboxes_shape[2])
    # tf.add_to_collection('xywh_raw', bboxes_logits)
    # tf.add_to_collection('xywh', tf.stack([x1s, y1s, ws, hs], axis=-1))
    x1s = (x1s - ws/2.0)
    y1s = (y1s - hs/2.0)
    x2s = (x1s + ws)
    y2s = (y1s + hs)
    # NOTE tf.image.non_max_suppression takes in the coordinates in this format.
    # bboxes has shape of [B, N, H, W, 4]
    bboxes = tf.stack([y1s, x1s, y2s, x2s], axis=-1, name='bboxes')

    # scale_logits has shape of [B, N, H, W, 1]
    scale_logits = tf.stack([logits[:, :, :, i*(FLAGS.num_classes+5) + 4:i*(FLAGS.num_classes+5) + 5] for i in xrange(FLAGS.num_bboxes)], 0)
    # tf.add_to_collection('scale_logits', scale_logits)

    # cls_logits has shape of [B, N, H, W, 80]
    cls_logits = tf.stack([logits[:, :, :, i*(FLAGS.num_classes+5) + 5:(i+1)*(FLAGS.num_classes + 5)] for i in xrange(FLAGS.num_bboxes)], 0)
    # tf.add_to_collection('cls_logits', cls_logits)

    # probabilities has shape of [B, N, H, W, 80]
    probabilities = cls_logits * scale_logits
    # tf.add_to_collection('probabilities', probabilities)

    # probabilities_filtered has shape of [B, N, H, W, 80]
    probs_shape = map(lambda f: int(f), probabilities.get_shape())
    probabilities_bool = tf.greater(probabilities, FLAGS.prob_thresh)
    probabilities_filtered = tf.where(probabilities_bool, probabilities, tf.zeros(probs_shape), name='probabilities')
    # tf.add_to_collection('probabilities_filtered', probabilities_filtered)
    return bboxes, probabilities_filtered
