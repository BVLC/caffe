# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from fast_rcnn.config import cfg
from fast_rcnn.train import filter_roidb
from utils.blob import im_list_to_blob
from utils.timer import Timer
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform
import numpy as np
import cv2

def _vis_proposals(im, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    class_name = 'obj'
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []

    assert len(cfg.TEST.SCALES) == 1
    target_size = cfg.TEST.SCALES[0]

    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_info = np.hstack((im.shape[:2], im_scale))[np.newaxis, :]
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_info

def im_proposals(net, im):
    """Generate RPN proposals on a single image."""
    blobs = {}
    blobs['data'], blobs['im_info'] = _get_image_blob(im)
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    blobs_out = net.forward(
            data=blobs['data'].astype(np.float32, copy=False),
            im_info=blobs['im_info'].astype(np.float32, copy=False))

    scale = blobs['im_info'][0, 2]
    boxes = blobs_out['rois'][:, 1:].copy() / scale
    scores = blobs_out['scores'].copy()
    return boxes, scores

def imdb_proposals(net, imdb):
    """Generate RPN proposals on all images in an imdb."""

    _t = Timer()
    imdb_boxes = [[] for _ in xrange(imdb.num_images)]
    for i in xrange(imdb.num_images):
        im = None
        if cfg.TRAIN.FORMAT == 'pickle':
            with open(imdb.image_path_at(i), 'rb') as f:
                im = cPickle.load(f)
        else:
            im = cv2.imread(imdb.image_path_at(i))

        _t.tic()
        imdb_boxes[i], scores = im_proposals(net, im)
        _t.toc()
        print 'im_proposals: {:d}/{:d} {:.3f}s' \
              .format(i + 1, imdb.num_images, _t.average_time)
        if 0:
            dets = np.hstack((imdb_boxes[i], scores))
            # from IPython import embed; embed()
            _vis_proposals(im, dets[:3, :], thresh=0.9)
            plt.show()

    return imdb_boxes


def imdb_rpn_compute_stats(net, imdb, anchor_scales=(8,16,32),
                           feature_stride=16):
    raw_anchors = generate_anchors(scales=np.array(anchor_scales))
    print raw_anchors.shape
    sums = 0
    squred_sums = 0
    counts = 0
    roidb = filter_roidb(imdb.roidb)
    # Compute a map of input image size and output feature map blob
    map_w = {}
    map_h = {}
    for i in xrange(50, cfg.TRAIN.MAX_SIZE + 10):
        blobs = {
            'data': np.zeros((1, 3, i, i)),
            'im_info': np.asarray([[i, i, 1.0]])
        }
        net.blobs['data'].reshape(*(blobs['data'].shape))
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
        blobs_out = net.forward(
            data=blobs['data'].astype(np.float32, copy=False),
            im_info=blobs['im_info'].astype(np.float32, copy=False))
        height, width = net.blobs['rpn/output'].data.shape[-2:]
        map_w[i] = width
        map_h[i] = height

    for i in xrange(len(roidb)):
        if not i % 5000:
            print 'computing %d/%d' % (i, imdb.num_images)
        im = None
        if cfg.TRAIN.FORMAT == 'pickle':
            with open(roidb[i]['image'], 'rb') as f:
                im = cPickle.load(f)
        else:
            im = cv2.imread(roidb[i]['image'])

        im_data, im_info = _get_image_blob(im)
        gt_boxes = roidb[i]['boxes']
        gt_boxes = gt_boxes * im_info[0, 2]
        height = map_h[im_data.shape[2]]
        width = map_w[im_data.shape[3]]
        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * feature_stride
        shift_y = np.arange(0, height) * feature_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = raw_anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = (raw_anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= 0) &
            (all_anchors[:, 1] >= 0) &
            (all_anchors[:, 2] < im_info[0, 1]) &  # width
            (all_anchors[:, 3] < im_info[0, 0])  # height
        )[0]

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))

        # There are 2 types of bbox targets
        # 1. anchor whose overlaps with gt is greater than RPN_POSITIVE_OVERLAP
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP)[0]
        # 2. anchors which best match certain gt
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        fg_inds = np.unique(np.hstack((fg_inds, gt_argmax_overlaps)))
        gt_rois = gt_boxes[argmax_overlaps, :]

        anchors = anchors[fg_inds, :]
        gt_rois = gt_rois[fg_inds, :]
        targets = bbox_transform(anchors, gt_rois[:, :4]).astype(np.float32, copy=False)
        sums += targets.sum(axis=0)
        squred_sums += (targets ** 2).sum(axis=0)
        counts += targets.shape[0]

    means = sums / counts
    stds = np.sqrt(squred_sums / counts - means ** 2)
    print means
    print stds
    return means, stds
