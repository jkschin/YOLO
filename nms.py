import numpy as np

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

def nms(bboxes, probabilities, iou_thresh):
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
