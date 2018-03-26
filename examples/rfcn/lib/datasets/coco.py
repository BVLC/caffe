# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from fast_rcnn.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import json
import uuid
# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask

def _filter_crowd_proposals(roidb, crowd_thresh):
    """
    Finds proposals that are inside crowd regions and marks them with
    overlap = -1 (for all gt rois), which means they will be excluded from
    training.
    """
    for ix, entry in enumerate(roidb):
        overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(overlaps.max(axis=1) == -1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        iscrowd = [int(True) for _ in xrange(len(crowd_inds))]
        crowd_boxes = ds_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = ds_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        overlaps[non_gt_inds[bad_inds], :] = -1
        roidb[ix]['gt_overlaps'] = scipy.sparse.csr_matrix(overlaps)
    return roidb

class coco(imdb):
    def __init__(self, image_set, year):
        imdb.__init__(self, 'coco_' + year + '_' + image_set)
        # COCO specific config options
        self.config = {'top_k' : 2000,
                       'use_salt' : True,
                       'cleanup' : True,
                       'crowd_thresh' : 0.7,
                       'rpn_file': None,
                       'min_size' : 2}
        # name, paths
        self._year = year
        self._image_set = image_set
        self._data_path = osp.join(cfg.DATA_DIR, 'coco')
        # load COCO API, classes, class <-> id mappings
        self._COCO = COCO(self._get_ann_file())
        cats = self._COCO.loadCats(self._COCO.getCatIds())
        self._classes = tuple(['__background__'] + [c['name'] for c in cats])
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._class_to_coco_cat_id = dict(zip([c['name'] for c in cats],
                                              self._COCO.getCatIds()))
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self.set_proposal_method('selective_search')
        self.competition_mode(False)

        # Some image sets are "views" (i.e. subsets) into others.
        # For example, minival2014 is a random 5000 image subset of val2014.
        # This mapping tells us where the view's images and proposals come from.
        self._view_map = {
            'minival2014' : 'val2014',          # 5k val2014 subset
            'valminusminival2014' : 'val2014',  # val2014 \setminus minival2014
        }
        coco_name = image_set + year  # e.g., "val2014"
        self._data_name = (self._view_map[coco_name]
                           if self._view_map.has_key(coco_name)
                           else coco_name)
        # Dataset splits that have ground-truth annotations (test splits
        # do not have gt annotations)
        self._gt_splits = ('train', 'val', 'minival')

    def _get_ann_file(self):
        prefix = 'instances' if self._image_set.find('test') == -1 \
                             else 'image_info'
        return osp.join(self._data_path, 'annotations',
                        prefix + '_' + self._image_set + self._year + '.json')

    def _load_image_set_index(self):
        """
        Load image ids.
        """
        image_ids = self._COCO.getImgIds()
        return image_ids

    def _get_widths(self):
        anns = self._COCO.loadImgs(self._image_index)
        widths = [ann['width'] for ann in anns]
        return widths

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = ('COCO_' + self._data_name + '_' +
                     str(index).zfill(12) + '.jpg')
        image_path = osp.join(self._data_path, 'images',
                              self._data_name, file_name)
        assert osp.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def selective_search_roidb(self):
        return self._roidb_from_proposals('selective_search')

    def edge_boxes_roidb(self):
        return self._roidb_from_proposals('edge_boxes_AR')

    def mcg_roidb(self):
        return self._roidb_from_proposals('MCG')

    def rpn_roidb(self):
        if (self._image_set != 'val') and ('test' not in self._image_set):
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _roidb_from_proposals(self, method):
        """
        Creates a roidb from pre-computed proposals of a particular methods.
        """
        top_k = self.config['top_k']
        cache_file = osp.join(self.cache_path, self.name +
                              '_{:s}_top{:d}'.format(method, top_k) +
                              '_roidb.pkl')

        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{:s} {:s} roidb loaded from {:s}'.format(self.name, method,
                                                            cache_file)
            return roidb

        if self._image_set in self._gt_splits:
            gt_roidb = self.gt_roidb()
            method_roidb = self._load_proposals(method, gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, method_roidb)
            # Make sure we don't use proposals that are contained in crowds
            roidb = _filter_crowd_proposals(roidb, self.config['crowd_thresh'])
        else:
            roidb = self._load_proposals(method, None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote {:s} roidb to {:s}'.format(method, cache_file)
        return roidb

    def _load_proposals(self, method, gt_roidb):
        """
        Load pre-computed proposals in the format provided by Jan Hosang:
        http://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-
          computing/research/object-recognition-and-scene-understanding/how-
          good-are-detection-proposals-really/
        For MCG, use boxes from http://www.eecs.berkeley.edu/Research/Projects/
          CS/vision/grouping/mcg/ and convert the file layout using
        lib/datasets/tools/mcg_munge.py.
        """
        box_list = []
        top_k = self.config['top_k']
        valid_methods = [
            'MCG',
            'selective_search',
            'edge_boxes_AR',
            'edge_boxes_70']
        assert method in valid_methods

        print 'Loading {} boxes'.format(method)
        for i, index in enumerate(self._image_index):
            if i % 1000 == 0:
                print '{:d} / {:d}'.format(i + 1, len(self._image_index))

            box_file = osp.join(
                cfg.DATA_DIR, 'coco_proposals', method, 'mat',
                self._get_box_file(index))

            raw_data = sio.loadmat(box_file)['boxes']
            boxes = np.maximum(raw_data - 1, 0).astype(np.uint16)
            if method == 'MCG':
                # Boxes from the MCG website are in (y1, x1, y2, x2) order
                boxes = boxes[:, (1, 0, 3, 2)]
            # Remove duplicate boxes and very small boxes and then take top k
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            boxes = boxes[:top_k, :]
            box_list.append(boxes)
            # Sanity check
            im_ann = self._COCO.loadImgs(index)[0]
            width = im_ann['width']
            height = im_ann['height']
            ds_utils.validate_boxes(boxes, width=width, height=height)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_coco_annotation(index)
                    for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        return gt_roidb

    def _load_coco_annotation(self, index):
        """
        Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        im_ann = self._COCO.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self._COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = self._COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Lookup table to map from COCO category ids to our internal class
        # indices
        coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                          self._class_to_ind[cls])
                                         for cls in self._classes[1:]])

        for ix, obj in enumerate(objs):
            cls = coco_cat_id_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                overlaps[ix, :] = -1.0
            else:
                overlaps[ix, cls] = 1.0

        ds_utils.validate_boxes(boxes, width=width, height=height)
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_box_file(self, index):
        # first 14 chars / first 22 chars / all chars + .mat
        # COCO_val2014_0/COCO_val2014_000000447/COCO_val2014_000000447991.mat
        file_name = ('COCO_' + self._data_name +
                     '_' + str(index).zfill(12) + '.mat')
        return osp.join(file_name[:14], file_name[:22], file_name)

    def _print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print ('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
               '~~~~').format(IoU_lo_thresh, IoU_hi_thresh)
        print '{:.1f}'.format(100 * ap_default)
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print '{:.1f}'.format(100 * ap)

        print '~~~~ Summary metrics ~~~~'
        coco_eval.summarize()

    def _do_detection_eval(self, res_file, output_dir):
        ann_type = 'bbox'
        coco_dt = self._COCO.loadRes(res_file)
        coco_eval = COCOeval(self._COCO, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_eval_metrics(coco_eval)
        eval_file = osp.join(output_dir, 'detection_results.pkl')
        with open(eval_file, 'wb') as fid:
            cPickle.dump(coco_eval, fid, cPickle.HIGHEST_PROTOCOL)
        print 'Wrote COCO eval results to: {}'.format(eval_file)

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_index):
            dets = boxes[im_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
              [{'image_id' : index,
                'category_id' : cat_id,
                'bbox' : [xs[k], ys[k], ws[k], hs[k]],
                'score' : scores[k]} for k in xrange(dets.shape[0])])
        return results

    def _write_coco_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                          self.num_classes - 1)
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind],
                                                           coco_cat_id))
        print 'Writing results json to {}'.format(res_file)
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = osp.join(output_dir, ('detections_' +
                                         self._image_set +
                                         self._year +
                                         '_results'))
        if self.config['use_salt']:
            res_file += '_{}'.format(str(uuid.uuid4()))
        res_file += '.json'
        self._write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self._image_set.find('test') == -1:
            self._do_detection_eval(res_file, output_dir)
        # Optionally cleanup results json file
        if self.config['cleanup']:
            os.remove(res_file)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True
