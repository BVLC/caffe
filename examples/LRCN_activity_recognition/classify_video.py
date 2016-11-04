#!/bin/env python
#classify_video.py will classify a video using (1) singleFrame RGB model (2) singleFrame flow model (3) 0.5/0.5 singleFrame RGB/singleFrame flow fusion (4) 0.33/0.67 singleFrame RGB/singleFrame flow fusion (5) LRCN RGB model (6) LRCN flow model (7) 0.5/0.5 LRCN RGB/LRCN flow model (8) 0.33/0.67 LRCN RGB/LRCN flow model
#Before using, change RGB_video_path and flow_video_path.
#Use: classify_video.py video, where video is the video you wish to classify.  If no video is specified, the video "v_Archery_g01_c01" will be classified.

import numpy as np
import glob
caffe_root = '../../'
import sys
sys.path.insert(0,caffe_root + 'python')
import caffe
caffe.set_mode_cpu()
import pickle

flow_video_path = 'flow_images/'
if len(sys.argv) > 1:
  video = sys.argv[1]
  if len(sys.argv) > 2:
    RGB_video_path = sys.argv[2]
  else:
    RGB_video_path = 'frames/'
else:
  video = 'v_Archery_g01_c01'

#Initialize transformers

def initialize_transformer(image_mean):
  shape = (10*16, 3, 227, 227)
  transformer = caffe.io.Transformer({'data': shape})
  channel_mean = np.zeros((3,227,227))
  for channel_index, mean_val in enumerate(image_mean):
    channel_mean[channel_index, ...] = mean_val
  transformer.set_mean('data', channel_mean)
  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2, 1, 0))
  transformer.set_transpose('data', (2, 0, 1))
  return transformer


ucf_mean_RGB = np.zeros((3,1,1))
ucf_mean_flow = np.zeros((3,1,1))
ucf_mean_flow[:,:,:] = 128
ucf_mean_RGB[0,:,:] = 103.939
ucf_mean_RGB[1,:,:] = 116.779
ucf_mean_RGB[2,:,:] = 128.68

transformer_RGB = initialize_transformer(ucf_mean_RGB)

# Extract list of frames in video
RGB_frames = glob.glob('%s%s/*.jpg' %(RGB_video_path, video))
flow_frames = glob.glob('%s%s/*.jpg' %(flow_video_path, video))

#classify video with LRCN model
def LRCN_classify_video(frames, net, transformer):
  clip_length = 16
  offset = 8
  input_images = []
  for im in frames:
    input_im = caffe.io.load_image(im)
    if (input_im.shape[0] < 240):
      input_im = caffe.io.resize_image(input_im, (240,320))
    input_images.append(input_im)
  vid_length = len(input_images)
  input_data = []
  for i in range(0,vid_length,offset):
    if (i + clip_length) < vid_length:
      input_data.extend(input_images[i:i+clip_length])
    else:  #video may not be divisible by clip_length
      input_data.extend(input_images[-clip_length:])
  output_predictions = np.zeros((len(input_data),101))
  for i in range(0,len(input_data),clip_length):
    clip_input = input_data[i:i+clip_length]
    clip_input = caffe.io.oversample(clip_input,[227,227])
    clip_clip_markers = np.ones((clip_input.shape[0],1,1,1))
    clip_clip_markers[0:10,:,:,:] = 0
#    if is_flow:  #need to negate the values when mirroring
#      clip_input[5:,:,:,0] = 1 - clip_input[5:,:,:,0]
    caffe_in = np.zeros(np.array(clip_input.shape)[[0,3,1,2]], dtype=np.float32)
    for ix, inputs in enumerate(clip_input):
      caffe_in[ix] = transformer.preprocess('data',inputs)
    out = net.forward_all(data=caffe_in, clip_markers=np.array(clip_clip_markers))
    output_predictions[i:i+clip_length] = np.mean(out['probs'],1)
  return np.mean(output_predictions,0).argmax(), output_predictions

#classify video with singleFrame model
def singleFrame_classify_video(frames, net, transformer, is_flow):
  batch_size = 16 
  input_images = []
  for im in frames:
    input_im = caffe.io.load_image(im)
    if (input_im.shape[0] < 240):
      input_im = caffe.io.resize_image(input_im, (240,320))
    input_images.append(input_im)
  vid_length = len(input_images)

  output_predictions = np.zeros((len(input_images),101))
  for i in range(0,len(input_images), batch_size):
    clip_input = input_images[i:min(i+batch_size, len(input_images))]
    clip_input = caffe.io.oversample(clip_input,[227,227])
    clip_clip_markers = np.ones((clip_input.shape[0],1,1,1))
    clip_clip_markers[0:10,:,:,:] = 0
    if is_flow:  #need to negate the values when mirroring
      clip_input[5:,:,:,0] = 1 - clip_input[5:,:,:,0]
    caffe_in = np.zeros(np.array(clip_input.shape)[[0,3,1,2]], dtype=np.float32)
    for ix, inputs in enumerate(clip_input):
      caffe_in[ix] = transformer.preprocess('data',inputs)
    net.blobs['data'].reshape(caffe_in.shape[0], caffe_in.shape[1], caffe_in.shape[2], caffe_in.shape[3])
    out = net.forward_all(data=caffe_in)
    output_predictions[i:i+batch_size] = np.mean(out['probs'].reshape(10,caffe_in.shape[0]/10,101),0)
  return np.mean(output_predictions,0).argmax(), output_predictions

#Models and weights
singleFrame_model = 'deploy_singleFrame.prototxt'
lstm_model = 'deploy_lstm.prototxt'
RGB_singleFrame = 'single_frame_all_layers_hyb_RGB_iter_5000.caffemodel'
flow_singleFrame = 'single_frame_all_layers_hyb_flow_iter_50000.caffemodel'
RGB_lstm = 'RGB_lstm_model_iter_30000.caffemodel'
flow_lstm = 'flow_lstm_model_iter_50000.caffemodel'

#RGB_singleFrame_net =  caffe.Net(singleFrame_model, RGB_singleFrame, caffe.TEST)
#class_RGB_singleFrame, predictions_RGB_singleFrame = \
         #singleFrame_classify_video(RGB_frames, RGB_singleFrame_net, transformer_RGB, False)
#del RGB_singleFrame_net

#flow_singleFrame_net =  caffe.Net(singleFrame_model, flow_singleFrame, caffe.TEST)
#class_flow_singleFrame, predictions_flow_singleFrame = \
         #singleFrame_classify_video(flow_frames, flow_singleFrame_net, transformer_flow, True)
#del flow_singleFrame_net

RGB_lstm_net =  caffe.Net(lstm_model, RGB_lstm, caffe.TEST)
class_RGB_LRCN, predictions_RGB_LRCN = \
         LRCN_classify_video(RGB_frames, RGB_lstm_net, transformer_RGB)
del RGB_lstm_net

#flow_lstm_net =  caffe.Net(lstm_model, flow_lstm, caffe.TEST)
#class_flow_LRCN, predictions_flow_LRCN = \
         #LRCN_classify_video(flow_frames, flow_lstm_net, transformer_flow, True)
#del flow_lstm_net

#def compute_fusion(RGB_pred, flow_pred, p):
  #return np.argmax(p*np.mean(RGB_pred,0) + (1-p)*np.mean(flow_pred,0))  

#Load activity label hash
action_hash = pickle.load(open('action_hash_rev.p','rb'))

#print "RGB single frame model classified video as: %s.\n" %(action_hash[class_RGB_singleFrame])
#print "Flow single frame model classified video as: %s.\n" %(action_hash[class_flow_singleFrame])
print "RGB LRCN model classified video as: %s.\n" %(action_hash[class_RGB_LRCN])
#print "Flow LRCN frame model classified video as: %s.\n" %(action_hash[class_flow_LRCN])
#print "0.5/0.5 single frame fusion model classified video as: %s. \n" %(action_hash[compute_fusion(predictions_RGB_singleFrame, predictions_flow_singleFrame, 0.5)])
#print "0.33/0.67 single frame fusion model classified video as: %s. \n" %(action_hash[compute_fusion(predictions_RGB_singleFrame, predictions_flow_singleFrame, 0.33)])
#print "0.5/0.5 LRCN fusion model classified video as: %s. \n" %(action_hash[compute_fusion(predictions_RGB_LRCN, predictions_flow_LRCN, 0.5)])
#print "0.33/0.67 LRCN fusion model classified video as: %s. \n" %(action_hash[compute_fusion(predictions_RGB_LRCN, predictions_flow_LRCN, 0.33)])






