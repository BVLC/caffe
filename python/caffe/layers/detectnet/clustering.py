import math

import cv2 as cv
import numpy as np

import caffe

MAX_BOXES = 50


class ClusterGroundtruth(caffe.Layer):
    """
    * converts ground truth labels from grid format to list
    * bottom[0] - coverage-label
        [ batch_size x num_classes x grid_sz_x x grid_sz_y ]
    * bottom[1] - bbox-label
        [batch_size x 4 x grid_sz_x x grid_sz_y (xl, yt, xr, yb)]
    * top [i] - list of groundtruth bbox for each class
        [ batch_size x max_bbox_per_image x 5 (xl, yt, xr, yb, 0)]

    Example prototxt definition:

    layer {
        type: 'Python'
        name: 'cluster_gt'
        # gt_bbox_list is a batch_size x MAX_BOXES x 5 blob
        top: 'gt_bbox_list-class0'
        top: 'gt_bbox_list-class1'
        bottom: 'coverage-label'
        bottom: 'bbox-label'
        python_param {
            module: 'caffe.layers.detectnet.clustering'
            layer: 'ClusterGroundtruth'
            # parameters - img_size_x, img_size_y, stride, num_classes
            param_str : '1248,352,16,2'
        }
        include: { phase: TEST }
    }
    """

    def setup(self, bottom, top):
        self.is_groundtruth = True
        try:
            plist = self.param_str.split(',')
            self.image_size_x = int(plist[0])
            self.image_size_y = int(plist[1])
            self.stride = int(plist[2])
            self.num_classes = int(plist[3]) if len(plist) > 3 else 1
        except ValueError:
            raise ValueError("Parameter string missing or data type is wrong!")
        if len(top) != self.num_classes:
            raise ValueError("Unexpected number of tops: %d != %d" % (len(top), self.num_classes))

    def reshape(self, bottom, top):
        n_images = bottom[0].data.shape[0]
        num_classes = bottom[0].data.shape[1]
        if num_classes != self.num_classes:
            raise ValueError("Unexpected number of classes: %d != %d, bottom[0] shape=%s" % (num_classes, self.num_classes, repr(bottom[0].data.shape)))
        for i in xrange(num_classes):
            # Assuming that max booxes per image are MAX_BOXES
            top[i].reshape(n_images, MAX_BOXES, 5)

    def forward(self, bottom, top):
        for i in xrange(self.num_classes):
            data0 = bottom[0].data[:,i:i+1,:,:]
            bbox = cluster(self, data0, bottom[1].data)
            top[i].data[...] = bbox

    def backward(self, top, propagate_down, bottom):
        pass


class ClusterDetections(caffe.Layer):
    """
    * convert network output in grid format to list using group rectangles clustering
    * bottom[0] - predicted coverage
        [ batch_size x num_classes x grid_sz_x x grid_sz_y ]
    * bottom[1] - predicted bbox
        [batch_size x grid_sz_x x grid_sz_y x 4 (xl, yt, xr, yb)]
    * top [i] - list of predicted bbox for each class
        [ batch_size x max_bbox_per_image x 5 (xl, yt, xr, yb, confidence) ]

    Example prototxt definition:

    layer {
        type: 'Python'
        name: 'cluster'
        # det_bbox_list is a batch_size x MAX_BOXES x 5 blob
        top: 'det_bbox_list-class0'
        top: 'det_bbox_list-class1'
        bottom: 'coverage'
        bottom: 'bbox/regressor'
        python_param {
            module: 'caffe.layers.detectnet.clustering'
            layer: 'ClusterDetections'
            # parameters - img_size_x, img_size_y, stride,
            # gridbox_cvg_threshold,gridbox_rect_threshold,gridbox_rect_eps,min_height,num_classes
            param_str : '1248,352,16,0.05,1,0.025,22,2'
        }
        include: { phase: TEST }
    }
    """

    def setup(self, bottom, top):
        self.is_groundtruth = False
        try:
            plist = self.param_str.split(',')
            self.image_size_x = int(plist[0])
            self.image_size_y = int(plist[1])
            self.stride = int(plist[2])
            self.gridbox_cvg_threshold = float(plist[3])
            self.gridbox_rect_thresh = int(plist[4])
            self.gridbox_rect_eps = float(plist[5])
            self.min_height = int(plist[6])
            self.num_classes = int(plist[7]) if len(plist) > 7 else 1
        except ValueError:
            raise ValueError("Parameter string missing or data type is wrong!")
        if len(top) != self.num_classes:
            raise ValueError("Unexpected number of tops: %d != %d" % (len(top), self.num_classes))

    def reshape(self, bottom, top):
        n_images = bottom[0].data.shape[0]
        num_classes = bottom[0].data.shape[1]
        if num_classes != self.num_classes:
            raise ValueError("Unexpected number of classes: %d != %d, bottom[0] shape=%s" % (num_classes, self.num_classes, repr(bottom[0].data.shape)))
        for i in xrange(num_classes):
            # Assuming that max booxes per image are MAX_BOXES
            top[i].reshape(n_images, MAX_BOXES, 5)

    def forward(self, bottom, top):
        for i in xrange(self.num_classes):
            data0 = bottom[0].data[:,i:i+1,:,:]
            bbox = cluster(self, data0, bottom[1].data)
            top[i].data[...] = bbox

    def backward(self, top, propagate_down, bottom):
        pass


def gridbox_to_boxes(net_cvg, net_boxes, self):
    im_sz_x = self.image_size_x
    im_sz_y = self.image_size_y
    stride = self.stride

    grid_sz_x = int(im_sz_x / stride)
    grid_sz_y = int(im_sz_y / stride)

    boxes = []
    cvgs = []

    cell_width = im_sz_x / grid_sz_x
    cell_height = im_sz_y / grid_sz_y

    cvg_val = net_cvg[0][0:grid_sz_y][0:grid_sz_x]

    if (self.is_groundtruth):
        mask = (cvg_val > 0)
    else:
        mask = (cvg_val >= self.gridbox_cvg_threshold)
    coord = np.where(mask == 1)

    y = np.asarray(coord[0])
    x = np.asarray(coord[1])

    mx = x * cell_width
    my = y * cell_height

    x1 = (np.asarray([net_boxes[0][y[i]][x[i]] for i in xrange(x.size)]) + mx)
    y1 = (np.asarray([net_boxes[1][y[i]][x[i]] for i in xrange(x.size)]) + my)
    x2 = (np.asarray([net_boxes[2][y[i]][x[i]] for i in xrange(x.size)]) + mx)
    y2 = (np.asarray([net_boxes[3][y[i]][x[i]] for i in xrange(x.size)]) + my)

    boxes = np.transpose(np.vstack((x1, y1, x2, y2)))
    cvgs = np.transpose(np.vstack((x, y, np.asarray(
        [cvg_val[y[i]][x[i]] for i in xrange(x.size)]))))
    return boxes, cvgs, mask


def vote_boxes(propose_boxes, propose_cvgs, mask, self):
    """ Vote amongst the boxes using openCV's built-in clustering routine.
    """

    detections_per_image = []
    if not propose_boxes.any():
        return detections_per_image

    ######################################################################
    # GROUP RECTANGLES Clustering
    ######################################################################
    nboxes, weights = cv.groupRectangles(
        np.array(propose_boxes).tolist(),
        self.gridbox_rect_thresh,
        self.gridbox_rect_eps)
    if len(nboxes):
        for rect, weight in zip(nboxes, weights):
            if (rect[3] - rect[1]) >= self.min_height:
                confidence = math.log(weight[0])
                detection = [rect[0], rect[1], rect[2], rect[3], confidence]
                detections_per_image.append(detection)

    return detections_per_image


def cluster(self, net_cvg, net_boxes):
    """
    Read output of inference and turn into Bounding Boxes
    """
    batch_size = net_cvg.shape[0]
    boxes = np.zeros([batch_size, MAX_BOXES, 5])

    for i in range(batch_size):

        cur_cvg = net_cvg[i]
        cur_boxes = net_boxes[i]

        if (self.is_groundtruth):
            # Gather proposals that pass a threshold -
            propose_boxes, propose_cvgs, mask = gridbox_to_boxes(
                cur_cvg, cur_boxes, self)
            # Remove duplicates from ground truth
            new_array = list({tuple(row) for row in propose_boxes})
            boxes_cur_image = np.asarray(new_array, dtype=np.float16)
        else:
            # Gather proposals that pass a threshold -
            propose_boxes, propose_cvgs, mask = gridbox_to_boxes(cur_cvg, cur_boxes, self)
            # Vote across the proposals to get bboxes
            boxes_cur_image = vote_boxes(propose_boxes, propose_cvgs, mask, self)
            boxes_cur_image = np.asarray(boxes_cur_image, dtype=np.float16)

        if (boxes_cur_image.shape[0] != 0):
            [r, c] = boxes_cur_image.shape
            boxes[i, 0:r, 0:c] = boxes_cur_image

    return boxes
