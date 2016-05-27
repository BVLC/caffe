import numpy as np

import caffe

MAX_BOXES = 50


class ScoreDetections(caffe.Layer):
    """
    * Marks up bbox predictions as true positive/ false positive and missed gtruth bbox as
        true negatives
    * bottom[0] - list of ground truth bbox
        [batch_size x max_bbox_per_image x 5 (xl, yt, xr, yb, 0) ]
    * bottom[1] - list of predicted bbox
        [batch_size x max_bbox_per_image x 5 (xl, yt, xr, yb, confidence) ]
    * top[0]- Marked up bbox
        [ batch_size x max_bbox_per_image x 5 (xl, yt, xr, yb, class)]
        class 1 - true positive, 2 - false positive, 3 - true negative

    Example prototxt definition:

    layer {
        type: 'Python'
        name: 'score'
        top: 'bbox_cl'
        bottom: 'gt_bbox_list'
        bottom: 'det_bbox_list'
        python_param {
            module: 'caffe.layers.detectnet.mean_ap'
            layer: 'ScoreDetections'
        }
        include: { phase: TEST }
    }
    """

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        n_images = bottom[0].data.shape[0]
        # Assuming that max booxes per image are MAX_BOXES
        top[0].reshape(n_images, MAX_BOXES, 5)
        assert(bottom[0].data.shape[0] == bottom[1].data.shape[0]), "# of images not matching!"

    def forward(self, bottom, top):
        bbox_list = score_det(bottom[0].data, bottom[1].data)
        top[0].data[...] = bbox_list

    def backward(self, top, propagate_down, bottom):
        pass


class mAP(caffe.Layer):
    """
    * calculates precision, recall, mean average precision from marked up bbox list
    * bottom[0] - marked up bbox list
        [ batch_size x max_bbox_per_image x 5(xl, yt, xr, yb, class) ]
    * top[0] - mAP
    * top[1] - precision
    * top[2] - recall

    Example prototxt definition:

    layer {
        type: 'Python'
        name: 'mAP'
        top: 'mAP'
        top: 'precision'
        top: 'recall'
        bottom: 'bbox_cl'
        python_param {
            module: 'caffe.layers.detectnet.mean_ap'
            layer: 'mAP'
            # parameters - img_size_x, img_size_y, stride
            param_str : '1248,352,16'
        }
        include: { phase: TEST }
    }
    """

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(1)
        self.false_positives = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.precision = 0
        self.recall = 0
        self.avp = 0

    def forward(self, bottom, top):
        calcmAP(bottom[0].data, self)
        top[0].data[...] = self.avp
        top[1].data[...] = self.precision
        top[2].data[...] = self.recall

    def backward(self, top, propagate_down, bottom):
        pass


def iou(det, rhs):
    x_overlap = max(0, min(det[2], rhs[2]) - max(det[0], rhs[0]))
    y_overlap = max(0, min(det[3], rhs[3]) - max(det[1], rhs[1]))
    overlap_area = x_overlap * y_overlap
    if overlap_area == 0:
        return 0
    det_area = (det[2]-det[0])*(det[3]-det[1])
    rhs_area = (rhs[2]-rhs[0])*(rhs[3]-rhs[1])
    unionarea = det_area + rhs_area - overlap_area
    return overlap_area/unionarea


def divide_zero_is_zero(a, b):
    return float(a)/float(b) if b != 0 else 0


def score_det(gt_bbox_list, det_bbox_list):
    threshold = 0.7
    matched_bbox = np.zeros([gt_bbox_list.shape[0], MAX_BOXES, 5])

    for k in range(gt_bbox_list.shape[0]):

        # Remove  zeros from detected bboxes
        cur_det_bbox = det_bbox_list[k, :, 0:4]
        cur_det_bbox = np.asarray(filter(lambda a: a.tolist() != [0, 0, 0, 0], cur_det_bbox))

        # Remove  zeros from label bboxes
        cur_gt_bbox = gt_bbox_list[k, :, 0:4]
        cur_gt_bbox = np.asarray(filter(lambda a: a.tolist() != [0, 0, 0, 0], cur_gt_bbox))

        gt_matched = np.zeros([cur_gt_bbox.shape[0]])
        det_matched = np.zeros([cur_det_bbox.shape[0]])

        for i in range(cur_gt_bbox.shape[0]):
            for j in range(cur_det_bbox.shape[0]):
                if (iou(cur_det_bbox[j], cur_gt_bbox[i]) >= threshold) and (det_matched[j] == 0):
                    gt_matched[i] = 1
                    det_matched[j] = 1
                    break

        tp = np.asarray([np.append(j, 1) for j in cur_det_bbox[np.where(det_matched == 1)]])
        fp = np.asarray([np.append(j, 2) for j in cur_det_bbox[np.where(det_matched == 0)]])
        tn = np.asarray([np.append(j, 3) for j in cur_gt_bbox[np.where(gt_matched == 0)]])

        temp = np.append(tp, fp)
        temp = np.append(temp, tn)
        temp = temp.reshape([temp.size/5, 5])
        matched_bbox[k, 0:temp.shape[0], :] = temp

    return matched_bbox


def calcmAP(scored_detections, self):
    self.true_positives = np.where(scored_detections[:, :, 4] == 1)[0].size
    self.false_positives = np.where(scored_detections[:, :, 4] == 2)[0].size
    self.true_negatives = np.where(scored_detections[:, :, 4] == 3)[0].size
    self.precision = divide_zero_is_zero(self.true_positives, self.true_positives+self.false_positives)*100.00
    self.recall = divide_zero_is_zero(self.true_positives, self.true_positives+self.true_negatives)*100.00
    self.avp = self.precision * self.recall / 100.0
