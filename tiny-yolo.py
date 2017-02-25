import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from matplotlib import pyplot as plt
from utils import *

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

load_binary = False
save = False
debug = True
num_classes = 80
num_bboxes = 5
anchor_priors = np.array([[0.738768, 0.874946], [2.42204, 2.65704], [4.30971, 7.04493], [10.246, 4.59428], [12.6868, 11.8741]])
coords = 4
prob_thresh = 0.25
iou_thresh = 0.8

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

    # NOTE Darknet padding method. No effect when using this.
    # d0_pad = int(conv_shape[0]/2)
    # d1_pad = int(conv_shape[1]/2)
    # inp_padded = tf.pad(inp, paddings=[[0,0], [d0_pad, d0_pad], [d1_pad, d1_pad], [0,0]])
    tf.add_to_collection('inp_debug', inp)
    conv = tf.nn.conv2d(inp, conv_weights, strides=[1, stride, stride, 1], padding='SAME')
    tf.add_to_collection('conv_debug', conv)
    if batch_norm:
        conv = tf.nn.batch_normalization(conv, mean_weights, variance_weights, offset=None, scale=scale_weights, variance_epsilon=0.00001)
    tf.add_to_collection('bn_debug', conv)
    conv_bias = tf.add(conv, bias_weights)
    tf.add_to_collection('bias_debug', conv_bias)
    return conv_bias if nonlinearity is None else nonlinearity(conv_bias, 0.1)

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
    for i in xrange(num_bboxes):
        base = i*(num_bboxes + num_classes)
        merge.append(logits[:, :, :, base:base+4])
        merge.append(tf.expand_dims(tf.sigmoid(logits[:, :, :, base+4]), -1))
        merge.append(tf.nn.softmax(logits[:, :, :, base+5:base+(num_classes+5)]))
    logits = tf.concat(3, merge)
    return logits

def process_logits(logits, class_feed):
    # bboxes_logits has shape of [B, N, H, W, 4]
    bboxes_logits = tf.stack([logits[:, :, :, i*(num_classes+5) + 0:i*(num_classes+5) + 4] for i in xrange(num_bboxes)], 0)
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
    tf.add_to_collection('xywh_raw', bboxes_logits)
    tf.add_to_collection('xywh', tf.stack([x1s, y1s, ws, hs], axis=-1))
    x1s = (x1s - ws/2.0)
    y1s = (y1s - hs/2.0)
    x2s = (x1s + ws)
    y2s = (y1s + hs)
    # NOTE tf.image.non_max_suppression takes in the coordinates in this format.
    # bboxes has shape of [B, N, H, W, 4]
    bboxes = tf.stack([y1s, x1s, y2s, x2s], axis=-1)

    # scale_logits has shape of [B, N, H, W, 1]
    scale_logits = tf.stack([logits[:, :, :, i*(num_classes+5) + 4:(i)*(num_classes+5) + 5] for i in xrange(num_bboxes)], 0)
    tf.add_to_collection('scale_logits', scale_logits)

    # cls_logits has shape of [B, N, H, W, 80]
    cls_logits = tf.stack([logits[:, :, :, i*(num_classes+5) + 5:(i+1)*(num_classes + 5)] for i in xrange(num_bboxes)], 0)
    tf.add_to_collection('cls_logits', cls_logits)

    # probabilities has shape of [B, N, H, W, 80]
    probabilities = cls_logits * scale_logits
    tf.add_to_collection('probabilities', probabilities)

    # probabilities_filtered has shape of [B, N, H, W, 80]
    probs_shape = map(lambda f: int(f), probabilities.get_shape())
    probabilities_bool = tf.greater(probabilities, prob_thresh)
    probabilities_filtered = tf.where(probabilities_bool, probabilities, tf.zeros(probs_shape))
    tf.add_to_collection('probabilities_filtered', probabilities_filtered)

    return bboxes, probabilities_filtered
    # car is index 2 so let's do that first
    # nms_boxes = tf.reshape(bboxes, [reduce(lambda x, y: x*y, bboxes_shape[:-1]), bboxes_shape[-1]])
    # nms_probs = tf.reshape(probabilities[:, :, :, :, class_feed], [reduce(lambda x, y: x*y, probs_shape[:-1])])
    # bboxes_indices = tf.image.non_max_suppression(nms_boxes, nms_probs, 100, iou_thresh)
    # bboxes_output = tf.gather(nms_boxes, bboxes_indices)

    # if debug:
    #     print 'bboxes_logits: ', bboxes_logits.get_shape()
    #     print 'row_grid_coords: ', row_grid_coords.get_shape()
    #     print 'col_grid_coords: ', col_grid_coords.get_shape()
    #     print 'x1s: ', x1s.get_shape()
    #     print 'y1s: ', y1s.get_shape()
    #     print 'x2s: ', x2s.get_shape()
    #     print 'y2s: ', y2s.get_shape()
    #     print 'bboxes: ', bboxes.get_shape()
    #     print 'scale_logits: ', scale_logits.get_shape()
    #     print 'cls_logits: ', cls_logits.get_shape()
    #     print 'probabilities: ', probabilities.get_shape()
    #     print 'nms_boxes: ', nms_boxes.get_shape()
    #     print 'nms_probs: ', nms_probs.get_shape()
    #     print 'bboxes_indices: ', bboxes_indices.get_shape()
    #     print 'bboxes_output: ', bboxes_output.get_shape()
    # return bboxes_output

def bbox_area(bbox):
    y1, x1, y2, x2 = bbox
    return (x2 - x1) * (y2 - y1)

def IOU(bbox1, bbox2):
    # format is y1 x1 y2 x2
    b1_y1, b1_x1, b1_y2, b1_x2 = bbox1
    b2_y1, b2_x1, b2_y2, b2_x2 = bbox2
    w = min(b1_x2, b2_x2) - max(b1_x1, b2_x1)
    h = min(b1_y2, b2_y2) - max(b1_y1, b2_y1)
    intersection = w*h
    union = bbox_area(bbox1) + bbox_area(bbox2) - intersection
    return intersection / float(union)

def nms(bboxes, probabilities):
    # bboxes has shape of [B, N, H, W, 4]
    # probabilities has shape of [B, N, H, W, 80]
    shape = bboxes.shape
    bboxes = np.reshape(bboxes, [reduce(lambda x, y: x*y, shape[:-1]), 4])
    probabilities = np.reshape(probabilities, [reduce(lambda x, y: x*y, shape[:-1]), 80])
    for i in xrange(bboxes.shape[0]):
        flag = False
        for k in xrange(80):
            if probabilities[i][k] > 0:
                flag = True
        if not flag:
            continue
        else:
            for j in xrange(i+1, bboxes.shape[0]):
                if IOU(bboxes[i], bboxes[j]) > iou_thresh:
                    for l in xrange(80):
                        if probabilities[i][l] < probabilities[j][l]:
                            probabilities[i][l] = 0.0
                        else:
                            probabilities[j][l] = 0.0
    bboxes_out = []
    classes = []
    for i in xrange(bboxes.shape[0]):
        for j in xrange(80):
            if probabilities[i][j] > 0.0:
                bboxes_out.append(bboxes[i])
                classes.append(j)
    return bboxes_out, classes

def main(argv):
    image = tf.placeholder(tf.float32, shape=[1, 416, 416, 3])
    original = tf.placeholder(tf.float32, shape=[1, 1080, 1920, 3])
    class_feed = tf.placeholder(tf.int32, shape=[])
    model = tf.make_template('model', model_spec)
    logits = model(image)
    bboxes, probabilities = process_logits(logits, class_feed)
    # image_bbox = tf.image.draw_bounding_boxes(original, tf.expand_dims(bboxes, 0))
    sess = tf.Session()
    saver = tf.train.Saver()
    if load_binary:
        sess.run(tf.global_variables_initializer())
        load_from_binary(sess)
    else:
        saver.restore(sess, '/home/jkschin/code/Tiny-YOLO/tiny-yolo-model.ckpt')

    # image_in = cv2.imread('/media/jkschin/WD2TB/data/20170206-lornie-road/7290_frames/290_0001.jpg').astype(np.float32)
    image_in = cv2.imread('sample.jpg').astype(np.float32)
    # YOLO original does processing in RGB
    image_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
    orig = image_in
    orig = np.expand_dims(orig, 0)
    orig = orig / 255.0
    image_in = cv2.resize(image_in, (416, 416))
    # image_in = darknet_resize(image_in, 416, 416)
    image_in = np.expand_dims(image_in, 0)
    # darknet scales values from 0 to 1
    image_in = (image_in / 255.0)
    class_feed_in = 2
    # logits_val = sess.run(logits, feed_dict={image:image_in, class_feed: class_feed_in, original:orig})
    # for idx, val in enumerate(logits_val.flatten()):
    #     print idx, val
    # print logits_val.shape
    # print_layer_weights(sess, feed_dict={image: image_in})
    # print_layer_outputs(sess, feed_dict={image: image_in, class_feed: class_feed_in, original:orig})
    print_xywh(sess, feed_dict={image: image_in, class_feed: class_feed_in, original:orig})
    print_xywh_raw(sess, feed_dict={image: image_in, class_feed: class_feed_in, original:orig})

    bboxes_val, probabilities_val = sess.run([bboxes, probabilities], feed_dict={image: image_in, class_feed: class_feed_in, original:orig})
    bboxes_out, classes = nms(bboxes_val, probabilities_val)
    orig *= 255.0
    orig = np.squeeze(orig)
    for i, box in enumerate(bboxes_out):
        scale_img = [1080, 1920, 1080, 1920]
        box = [int(a*b) for a,b in zip(box, scale_img)]
        if classes[i] == 2:
            cv2.rectangle(orig, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
        else:
            cv2.rectangle(orig, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), 2)
    cv2.imwrite('sample_out.jpg', orig)

#     for class_feed_in in xrange(2, 3):
#         logits_val, bboxes_val, image_bbox_val, bboxes_val = sess.run([logits, bboxes, image_bbox, bboxes], feed_dict={image: image_in, class_feed: class_feed_in, original:orig})
#         print bboxes_val
#         plt.imshow(np.squeeze(image_bbox_val))
#         plt.show()
    if save:
        saver.save(sess, '/home/jkschin/code/Tiny-YOLO/tiny-yolo-model.ckpt')
    # print bboxes_val
    # print probs_val[0][:, :, :, :, 2]


    # for i in range(13):
    #     for j in range(13):
    #         for k in range(425):
    #             print k+j*425+i*13*425, logits_val[i, j, k]
'''
Legacy test code that might still be needed
'''
    # probs_filtered = tf.get_collection('probabilities_filtered')
    # probs_filtered_val = sess.run(probs_filtered, feed_dict={image:image_in})[0]
    # for i in xrange(5):
    #     for j in xrange(1):
    #         for k in range(13):
    #             for l in range(13):
    #                 print i,j,k,l,probs_filtered_val[i,j,k,l,2]
    # scale_logits = tf.get_collection('scale_logits')
    # cls_logits = tf.get_collection('cls_logits')
    # scale_val = sess.run(scale_logits, feed_dict={image:image_in})[0]
    # cls_val = sess.run(cls_logits, feed_dict={image:image_in})[0]
    # for i in xrange(5):
    #     for j in xrange(1):
    #         for k in xrange(13):
    #             for l in xrange(13):
    #                 prob_obj = scale_val[i,j,k,l,0]
    #                 print i,j,k,l,prob_obj,cls_val[i,j,k,l,0:4]
    # probs = tf.get_collection('probabilities')
    # probs_val = sess.run(probs, feed_dict={image:image_in})[0]
    # for i in xrange(5):
    #     for j in xrange(1):
    #         for k in xrange(13):
    #             for l in xrange(13):
    #                 for m in xrange(80):
    #                     p = probs_val[i,j,k,l,m]
    #                     if p >= 0.001:
    #                         print i,j,k,l,m,probs_val[i,j,k,l,m]
    # print scale_logits_val
    # print scale_logits_val.shape


if __name__ == '__main__':
    # bbox1 = [0, 0, 100, 100]
    # bbox2 = [50, 50, 150, 150]
    # print bbox_area(bbox1)
    # print bbox_area(bbox2)
    # print IOU(bbox1, bbox2)
    # print IOU(bbox2, bbox1)
    tf.app.run()



