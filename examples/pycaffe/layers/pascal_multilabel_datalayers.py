# imports
import json, time, pickle, scipy.misc, skimage.io, caffe

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image

from pascal_multilabel_with_datalayer_tools import SimpleTransformer


class PascalMultilabelDataLayerSync(caffe.Layer):
    """
    This is a simple syncronous datalayer for training a multilabel model on PASCAL. 
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===
        
        # params is a python dictionary with layer parameters.
        params = eval(self.param_str) 

        # do some simple checks that we have the parameters we need.
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'split' in params.keys(), 'Params must include split (train, val, or test).'
        assert 'pascal_root' in params.keys(), 'Params must include pascal_root.'
        assert 'im_shape' in params.keys(), 'Params must include im_shape.'
 
        # store input as class variables
        self.batch_size = params['batch_size'] 
        self.im_shape = params['im_shape'] 
        self.pascal_root = params['pascal_root']
        self.im_shape = params['im_shape']
        self.indexlist = [line.rstrip('\n') for line in open(osp.join(self.pascal_root, 'ImageSets/Main', params['split'] + '.txt'))] #get list of image indexes.
        self._cur = 0 # current image
        self.transformer = SimpleTransformer() #this class does some simple data-manipulations

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, self.im_shape[0], self.im_shape[1]) # since we use a fixed input image size, we can shape the data layer once. Else, we'd have to do it in the reshape call.
        top[1].reshape(self.batch_size, 20)

        print "PascalMultilabelDataLayerSync initialized for split: {}, with bs:{}, im_shape:{}, and {} images.".format(params['split'], params['batch_size'], params['im_shape'], len(self.indexlist))


    def reshape(self, bottom, top):
        """ no need to reshape each time sine the input is fixed size (rows and columns) """
        pass 

    def forward(self, bottom, top):
        """
        Load data. 
        """
        for itt in range(self.batch_size):

            # Did we finish an epoch?
            if self._cur == len(self.indexlist):
                self._cur = 0
                shuffle(self.indexlist)
            
            # Load an image
            index = self.indexlist[self._cur] # Get the image index
            im = np.asarray(Image.open(osp.join(self.pascal_root, 'JPEGImages', index + '.jpg'))) # load image
            im = scipy.misc.imresize(im, self.im_shape) # resize
            
            # do a simple horizontal flip as data augmentation
            flip = np.random.choice(2)*2-1
            im = im[:, ::flip, :]

            # Load and prepare ground truth
            multilabel = np.zeros(20).astype(np.float32)
            anns = load_pascal_annotation(index, self.pascal_root)
            for label in anns['gt_classes']:
                # in the multilabel problem we don't care how MANY instances there are of each class. Only if they are present.
                multilabel[label - 1] = 1 # The "-1" is b/c we are not interested in the background class.

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = self.transformer.preprocess(im)
            top[1].data[itt, ...] = multilabel
            self._cur += 1

    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass




class PascalMultilabelDataLayerAsync(caffe.Layer):
    """
    This is a simple asyncronous datalayer for training a multilabel model on PASCAL. 
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===
        
        # params is a python dictionary with layer parameters.
        params = eval(self.param_str) 

        # do some simple checks that we have the parameters we need.
        assert 'batch_size' in params.keys(), 'Params must include batch size.'
        assert 'split' in params.keys(), 'Params must include split (train, val, or test).'
        assert 'pascal_root' in params.keys(), 'Params must include pascal_root.'
        assert 'im_shape' in params.keys(), 'Params must include im_shape.'

        self.batch_size = params['batch_size'] # we need to store this as a local variable.

        # === We are going to do the actual data processing in a seperate, helperclass, called BatchAdvancer. So let's forward the parame to that class ===
        self.thread_result = {}
        self.thread = None
        self.batch_advancer = BatchAdvancer(self.thread_result, params)
        self.dispatch_worker() # Let it start fetching data right away.

        # === reshape tops ===
        top[0].reshape(self.batch_size, 3, params['im_shape'][0], params['im_shape'][1]) # since we use a fixed input image size, we can shape the data layer once. Else, we'd have to do it in the reshape call.
        top[1].reshape(self.batch_size, 20) # Note the 20 channels (because PASCAL has 20 classes.)

        print "PascalMultilabelDataLayerAsync initialized for split: {}, with bs:{}, im_shape:{}.".format(params['split'], params['batch_size'], params['im_shape'])



    def reshape(self, bottom, top):
        """ no need to reshape each time sine the input is fixed size (rows and columns) """
        pass 

    def forward(self, bottom, top):
        """ this is the forward pass, where we load the data into the blobs. Since we run the BatchAdvance asynchronously, we just wait for it, and then copy """

        if self.thread is not None:
            self.join_worker() # wait until it is done.

        for top_index, name in zip(range(len(top)), self.top_names):
            for i in range(self.batch_size):
                top[top_index].data[i, ...] = self.thread_result[name][i] #Copy the already-prepared data to caffe.
        
        self.dispatch_worker() # let's go again while the GPU process this batch.

    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None

    def backward(self, top, propagate_down, bottom):
        """ this layer does not back propagate """
        pass


class BatchAdvancer():
    """
    This is the class that is run asynchronously and actually does the work.
    """
    def __init__(self, result, params):
        self.result = result
        self.batch_size = params['batch_size'] 
        self.im_shape = params['im_shape'] 
        self.pascal_root = params['pascal_root']
        self.im_shape = params['im_shape']
        self.indexlist = [line.rstrip('\n') for line in open(osp.join(self.pascal_root, 'ImageSets/Main', params['split'] + '.txt'))] #get list of image indexes.
        self._cur = 0 # current image
        self.transformer = SimpleTransformer() #this class does some simple data-manipulations

        print "BatchAdvancer initialized with {} images".format(len(self.indexlist))

    def __call__(self):
        """
        This does the same stuff as the forward layer of the synchronous layer. Exept that we store the data and labels in the result dictionary (as lists of length batchsize).
        """
        self.result['data'] = []
        self.result['label'] = []
        for itt in range(self.batch_size):

            # Did we finish an epoch?
            if self._cur == len(self.indexlist):
                self._cur = 0
                shuffle(self.indexlist)
            
            # Load an image
            index = self.indexlist[self._cur] # Get the image index
            im = np.asarray(Image.open(osp.join(self.pascal_root, 'JPEGImages', index + '.jpg'))) # load image
            im = scipy.misc.imresize(im, self.im_shape) # resize
            
            # do a simple horizontal flip as data augmentation
            flip = np.random.choice(2)*2-1
            im = im[:, ::flip, :]

            # Load and prepare ground truth
            multilabel = np.zeros(20).astype(np.float32)
            anns = load_pascal_annotation(index, self.pascal_root)
            for label in anns['gt_classes']:
                # in the multilabel problem we don't care how MANY instances there are of each class. Only if they are present.
                multilabel[label - 1] = 1 # The "-1" is b/c we are not interested in the background class.

            # Store in a result list.
            self.result['data'].append(self.transformer.preprocess(im))
            self.result['label'].append(multilabel)
            self._cur += 1


def load_pascal_annotation(index, pascal_root):
    """
    This code is borrowed from Ross Girshick's FAST-RCNN code (https://github.com/rbgirshick/fast-rcnn). It parses the PASCAL .xml metadata files. See publication for further details: (http://arxiv.org/abs/1504.08083).

    Thanks Ross!

    """
    classes = ('__background__', # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(classes, xrange(21)))

    filename = osp.join(pascal_root, 'Annotations', index + '.xml')
    # print 'Loading: {}'.format(filename)
    def get_data_from_tag(node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data

    with open(filename) as f:
        data = minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, 21), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        # Make pixel indexes 0-based
        x1 = float(get_data_from_tag(obj, 'xmin')) - 1
        y1 = float(get_data_from_tag(obj, 'ymin')) - 1
        x2 = float(get_data_from_tag(obj, 'xmax')) - 1
        y2 = float(get_data_from_tag(obj, 'ymax')) - 1
        cls = class_to_ind[
                str(get_data_from_tag(obj, "name")).lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes' : boxes,
            'gt_classes': gt_classes,
            'gt_overlaps' : overlaps,
            'flipped' : False,
            'index': index}

