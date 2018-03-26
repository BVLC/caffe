__author__ = 'tylin'
__version__ = '1.0.1'
# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  segToMask  - Convert polygon segmentation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>segToMask, COCO>showAnns

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from skimage.draw import polygon
import urllib
import copy
import itertools
import mask
import os

class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset = {}
        self.anns = []
        self.imgToAnns = {}
        self.catToImgs = {}
        self.imgs = {}
        self.cats = {}
        if not annotation_file == None:
            print 'loading annotations into memory...'
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            print 'Done (t=%0.2fs)'%(time.time()- tic)
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print 'creating index...'
        anns = {}
        imgToAnns = {}
        catToImgs = {}
        cats = {}
        imgs = {}
        if 'annotations' in self.dataset:
            imgToAnns = {ann['image_id']: [] for ann in self.dataset['annotations']}
            anns =      {ann['id']:       [] for ann in self.dataset['annotations']}
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']] += [ann]
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            imgs      = {im['id']: {} for im in self.dataset['images']}
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            cats = {cat['id']: [] for cat in self.dataset['categories']}
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat
            catToImgs = {cat['id']: [] for cat in self.dataset['categories']}
            if 'annotations' in self.dataset:
                for ann in self.dataset['annotations']:
                    catToImgs[ann['category_id']] += [ann['image_id']]

        print 'index created!'

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print '%s: %s'%(key, value)

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                # this can be changed by defaultdict
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if type(catNms) == list else [catNms]
        supNms = supNms if type(supNms) == list else [supNms]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if type(ids) == list:
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if type(ids) == list:
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def showAnns(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        if datasetType == 'instances':
            ax = plt.gca()
            polygons = []
            color = []
            for ann in anns:
                c = np.random.random((1, 3)).tolist()[0]
                if type(ann['segmentation']) == list:
                    # polygon
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((len(seg)/2, 2))
                        polygons.append(Polygon(poly, True,alpha=0.4))
                        color.append(c)
                else:
                    # mask
                    t = self.imgs[ann['image_id']]
                    if type(ann['segmentation']['counts']) == list:
                        rle = mask.frPyObjects([ann['segmentation']], t['height'], t['width'])
                    else:
                        rle = [ann['segmentation']]
                    m = mask.decode(rle)
                    img = np.ones( (m.shape[0], m.shape[1], 3) )
                    if ann['iscrowd'] == 1:
                        color_mask = np.array([2.0,166.0,101.0])/255
                    if ann['iscrowd'] == 0:
                        color_mask = np.random.random((1, 3)).tolist()[0]
                    for i in range(3):
                        img[:,:,i] = color_mask[i]
                    ax.imshow(np.dstack( (img, m*0.5) ))
            p = PatchCollection(polygons, facecolors=color, edgecolors=(0,0,0,1), linewidths=3, alpha=0.4)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print ann['caption']

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]
        # res.dataset['info'] = copy.deepcopy(self.dataset['info'])
        # res.dataset['licenses'] = copy.deepcopy(self.dataset['licenses'])

        print 'Loading and preparing results...     '
        tic = time.time()
        anns    = json.load(open(resFile))
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id+1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = mask.area([ann['segmentation']])[0]
                if not 'bbox' in ann:
                    ann['bbox'] = mask.toBbox([ann['segmentation']])[0]
                ann['id'] = id+1
                ann['iscrowd'] = 0
        print 'DONE (t=%0.2fs)'%(time.time()- tic)

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def download( self, tarDir = None, imgIds = [] ):
        '''
        Download COCO images from mscoco.org server.
        :param tarDir (str): COCO results directory name
               imgIds (list): images to be downloaded
        :return:
        '''
        if tarDir is None:
            print 'Please specify target directory'
            return -1
        if len(imgIds) == 0:
            imgs = self.imgs.values()
        else:
            imgs = self.loadImgs(imgIds)
        N = len(imgs)
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)
        for i, img in enumerate(imgs):
            tic = time.time()
            fname = os.path.join(tarDir, img['file_name'])
            if not os.path.exists(fname):
                urllib.urlretrieve(img['coco_url'], fname)
            print 'downloaded %d/%d images (t=%.1fs)'%(i, N, time.time()- tic)
