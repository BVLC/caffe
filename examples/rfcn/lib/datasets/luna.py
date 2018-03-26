# --------------------------------------------------------
# LUNA dataset reader
# YAO Matrix (yaoweifeng0301@126.com)
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
import gzip
from fast_rcnn.config import cfg

class luna(imdb):
    def __init__(self, image_set, year, data_path=None):
        imdb.__init__(self, 'luna_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._data_path = self._get_default_path() if data_path is None \
                            else data_path
        self._classes = ('__background__', # always index 0
                         'nodule')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.pickle.gz'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # LUNA specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._data_path), \
                'LUNA data path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        bb_directory = os.path.join(self._data_path, '1_1_1mm_slices_lung')
        [subfolder, filename] = index.split('-')
        image_path = os.path.join(bb_directory, 'subset' + str(subfolder), filename)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where LUNA is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'LUNA' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_luna_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
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

    def _load_luna_annotation(self, index):
        """
        Load image and bounding boxes info from luna bbox file format
        """
        bb_directory = os.path.join(self._data_path, '1_1_1mm_slices_bbox')
        [subfolder, filename] = index.split('-')
        abs_filename = os.path.join(bb_directory, 'subset' + str(subfolder), filename)

        with gzip.open(abs_filename, 'rb') as f:
            [gt_boxes, im_info] = cPickle.load(f)
            num_objs = len(gt_boxes)

            boxes = np.zeros((num_objs, 4), dtype = np.uint16)
            gt_classes = np.zeros((num_objs), dtype = np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype = np.float32)
            # "Seg" area for luna is just the box area
            seg_areas = np.zeros((num_objs), dtype = np.float32)

            # Load object bounding boxes into a data frame.
            for ix, obj in enumerate(gt_boxes):
                # Make pixel indexes 0-based
                x1 = obj[0]
                y1 = obj[1]
                x2 = obj[2]
                y2 = obj[3]
                cls = int(obj[4])
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
                seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

            overlaps = scipy.sparse.csr_matrix(overlaps)

            return {'boxes' : boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps' : overlaps,
                    'flipped' : False,
                    'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt'] else self._comp_id)
        return comp_id
    
if __name__ == '__main__':
    from datasets.luna import luna
    d = luna('trainval', '2016')
    res = d.roidb
    from IPython import embed; embed()
