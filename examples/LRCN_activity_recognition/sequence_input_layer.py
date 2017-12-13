#!/usr/bin/env python

#Data layer for video.  Change flow_frames and RGB_frames to be the path to the flow and RGB frames.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('../../python')
import caffe
import io
from PIL import Image
import numpy as np
import scipy.misc
import time
import pdb
import glob
import pickle as pkl
import random
import h5py
from multiprocessing import Pool
from threading import Thread
import skimage.io
import copy

flow_frames = '/home/link/flow_images/'
RGB_frames = '/home/link/frames/'
test_frames = 16 
train_frames = 16
test_buffer = 3
train_buffer = 24

def processImageCrop(im_info, transformer, flow):
  im_path = im_info[0]
  im_crop = im_info[1] 
  im_reshape = im_info[2]
  im_flip = im_info[3]
  data_in = caffe.io.load_image(im_path)
  if (data_in.shape[0] < im_reshape[0]) | (data_in.shape[1] < im_reshape[1]):
    data_in = caffe.io.resize_image(data_in, im_reshape)
  if im_flip:
    data_in = caffe.io.flip_image(data_in, 1, flow) 
    data_in = data_in[im_crop[0]:im_crop[2], im_crop[1]:im_crop[3], :] 
  processed_image = transformer.preprocess('data_in',data_in)
  return processed_image

class ImageProcessorCrop(object):
  def __init__(self, transformer, flow):
    self.transformer = transformer
    self.flow = flow
  def __call__(self, im_info):
    return processImageCrop(im_info, self.transformer, self.flow)

class sequenceGeneratorVideo(object):
  def __init__(self, buffer_size, clip_length, num_videos, video_dict, video_order):
    self.buffer_size = buffer_size
    self.clip_length = clip_length
    self.N = self.buffer_size*self.clip_length
    self.num_videos = num_videos
    self.video_dict = video_dict
    self.video_order = video_order
    self.idx = 0

  def __call__(self):
    label_r = []
    im_paths = []
    im_crop = []
    im_reshape = []  
    im_flip = []
 
    if self.idx + self.buffer_size >= self.num_videos:
      idx_list = range(self.idx, self.num_videos)
      idx_list.extend(range(0, self.buffer_size-(self.num_videos-self.idx)))
    else:
      idx_list = range(self.idx, self.idx+self.buffer_size)
    

    for i in idx_list:
      key = self.video_order[i]
      label = self.video_dict[key]['label']
      video_reshape = self.video_dict[key]['reshape']
      video_crop = self.video_dict[key]['crop']
      label_r.extend([label]*self.clip_length)

      im_reshape.extend([(video_reshape)]*self.clip_length)
      r0 = int(random.random()*(video_reshape[0] - video_crop[0]))
      r1 = int(random.random()*(video_reshape[1] - video_crop[1]))
      im_crop.extend([(r0, r1, r0+video_crop[0], r1+video_crop[1])]*self.clip_length)     
      f = random.randint(0,1)
      im_flip.extend([f]*self.clip_length)
      rand_frame = int(random.random()*(self.video_dict[key]['num_frames']-self.clip_length)+1+1)
      frames = []

      for i in range(rand_frame,rand_frame+self.clip_length):
        frames.append(self.video_dict[key]['frames'] %i)
     
      im_paths.extend(frames) 
    
    
    im_info = zip(im_paths,im_crop, im_reshape, im_flip)

    self.idx += self.buffer_size
    if self.idx >= self.num_videos:
      self.idx = self.idx - self.num_videos

    return label_r, im_info
  
def advance_batch(result, sequence_generator, image_processor, pool):
  
    label_r, im_info = sequence_generator()
    tmp = image_processor(im_info[0])
    result['data'] = pool.map(image_processor, im_info)
    result['label'] = label_r
    cm = np.ones(len(label_r))
    cm[0::16] = 0
    result['clip_markers'] = cm

class BatchAdvancer():
    def __init__(self, result, sequence_generator, image_processor, pool):
      self.result = result
      self.sequence_generator = sequence_generator
      self.image_processor = image_processor
      self.pool = pool
 
    def __call__(self):
      return advance_batch(self.result, self.sequence_generator, self.image_processor, self.pool)

class videoRead(caffe.Layer):

  def initialize(self):
    self.train_or_test = 'test'
    self.flow = False
    self.buffer_size = test_buffer  #num videos processed per batch
    self.frames = test_frames   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = 3
    self.height = 227
    self.width = 227
    self.path_to_images = RGB_frames 
    self.video_list = 'ucf101_split1_testVideos.txt' 

  def setup(self, bottom, top):
    random.seed(10)
    self.initialize()
    f = open(self.video_list, 'r')
    f_lines = f.readlines()
    f.close()

    video_dict = {}
    current_line = 0
    self.video_order = []
    for ix, line in enumerate(f_lines):
      video = line.split(' ')[0].split('/')[1]
      l = int(line.split(' ')[1])
      frames = glob.glob('%s%s/*.jpg' %(self.path_to_images, video))
      num_frames = len(frames)
      video_dict[video] = {}
      video_dict[video]['frames'] = frames[0].split('.')[0] + '.%04d.jpg'
      video_dict[video]['reshape'] = (240,320)
      video_dict[video]['crop'] = (227, 227)
      video_dict[video]['num_frames'] = num_frames
      video_dict[video]['label'] = l
      self.video_order.append(video) 

    self.video_dict = video_dict
    self.num_videos = len(video_dict.keys())

    #set up data transformer
    shape = (self.N, self.channels, self.height, self.width)
        
    self.transformer = caffe.io.Transformer({'data_in': shape})
    self.transformer.set_raw_scale('data_in', 255)
    if self.flow:
      image_mean = [128, 128, 128]
# TODO: No flow support in transformer currently
      self.transformer.set_is_flow('data_in', True)
    else:
      image_mean = [103.939, 116.779, 128.68]
# TODO: No flow support in transformer currently
      self.transformer.set_is_flow('data_in', False)
    channel_mean = np.zeros((3,227,227))
    for channel_index, mean_val in enumerate(image_mean):
      channel_mean[channel_index, ...] = mean_val
    self.transformer.set_mean('data_in', channel_mean)
    self.transformer.set_channel_swap('data_in', (2, 1, 0))
    self.transformer.set_transpose('data_in', (2, 0, 1))

    self.thread_result = {}
    self.thread = None
    pool_size = 24

    self.image_processor = ImageProcessorCrop(self.transformer, self.flow)
    self.sequence_generator = sequenceGeneratorVideo(self.buffer_size, self.frames, self.num_videos, self.video_dict, self.video_order)

    self.pool = Pool(processes=pool_size)
    self.batch_advancer = BatchAdvancer(self.thread_result, self.sequence_generator, self.image_processor, self.pool)
    self.dispatch_worker()
    self.top_names = ['data', 'label','clip_markers']
    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    self.join_worker()
    for top_index, name in enumerate(self.top_names):
      if name == 'data':
        shape = (self.N, self.channels, self.height, self.width)
      elif name == 'label':
        shape = (self.N,)
      elif name == 'clip_markers':
        shape = (self.N,)
      top[top_index].reshape(*shape)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
  
    if self.thread is not None:
      self.join_worker() 

    #rearrange the data: The LSTM takes inputs as [video0_frame0, video1_frame0,...] but the data is currently arranged as [video0_frame0, video0_frame1, ...]
    new_result_data = [None]*len(self.thread_result['data']) 
    new_result_label = [None]*len(self.thread_result['label']) 
    new_result_cm = [None]*len(self.thread_result['clip_markers'])
    for i in range(self.frames):
      for ii in range(self.buffer_size):
        old_idx = ii*self.frames + i
        new_idx = i*self.buffer_size + ii
        new_result_data[new_idx] = self.thread_result['data'][old_idx]
        new_result_label[new_idx] = self.thread_result['label'][old_idx]
        new_result_cm[new_idx] = self.thread_result['clip_markers'][old_idx]

    for top_index, name in zip(range(len(top)), self.top_names):
      if name == 'data':
        for i in range(self.N):
          top[top_index].data[i, ...] = new_result_data[i] 
      elif name == 'label':
        top[top_index].data[...] = new_result_label
      elif name == 'clip_markers':
        top[top_index].data[...] = new_result_cm

    self.dispatch_worker()
      
  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_advancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None

  def backward(self, top, propagate_down, bottom):
    pass

class videoReadTrain_flow(videoRead):

  def initialize(self):
    self.train_or_test = 'train'
    self.flow = True
    self.buffer_size = train_buffer  #num videos processed per batch
    self.frames = train_frames   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = 3
    self.height = 227
    self.width = 227
    self.path_to_images = flow_frames 
    self.video_list = 'ucf101_split1_trainVideos.txt' 

class videoReadTest_flow(videoRead):

  def initialize(self):
    self.train_or_test = 'test'
    self.flow = True
    self.buffer_size = test_buffer  #num videos processed per batch
    self.frames = test_frames   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = 3
    self.height = 227
    self.width = 227
    self.path_to_images = flow_frames 
    self.video_list = 'ucf101_split1_testVideos.txt' 

class videoReadTrain_RGB(videoRead):

  def initialize(self):
    self.train_or_test = 'train'
    self.flow = False
    self.buffer_size = train_buffer  #num videos processed per batch
    self.frames = train_frames   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = 3
    self.height = 227
    self.width = 227
    self.path_to_images = RGB_frames 
    self.video_list = 'ucf101_split1_trainVideos.txt' 

class videoReadTest_RGB(videoRead):

  def initialize(self):
    self.train_or_test = 'test'
    self.flow = False
    self.buffer_size = test_buffer  #num videos processed per batch
    self.frames = test_frames   #length of processed clip
    self.N = self.buffer_size*self.frames
    self.idx = 0
    self.channels = 3
    self.height = 227
    self.width = 227
    self.path_to_images = RGB_frames 
    self.video_list = 'ucf101_split1_testVideos.txt' 
