# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:25:18 2016

@author: denitome
"""

# TODO: remember to remove 0.5 (mean) from the images

import os
import re
import caffe
import cv2
import json
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# general settings
samplingRate = 5
offset = 25
inputSizeNN = 368
joints_idx = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
sigma = 7
stride = 8
verbose = True

def filterJoints(joints_orig):
    joints = [0] * len(joints_idx)
    for j in range(len(joints_idx)):
        joints[j] = map(int, joints_orig[joints_idx[j]])
    return joints
    
def getBoundingBox(data):
    joints = filterJoints(data['joint_self'])
            
    max_x = -1
    max_y = -1
    center = map(int, data['objpos'])
    for i in range(len(joints)):
        j = joints[i]
        if (max_x < abs(j[0]-center[0])):
            max_x = abs(j[0]-center[0])
        if (max_y < abs(j[1]-center[1])):
            max_y = abs(j[1]-center[1])
    offset_x = max_x + offset
    offset_y = max_y + offset
    if (offset_x > offset_y):
        offset_y = offset_x
    else:
        offset_x = offset_y
    if (center[0] + offset_x > data['img_width']):
        offset_x = data['img_width'] - center[0]
    if (center[0] - offset_x < 0):
        offset_x = center[0]
    if (center[1] + offset_y > data['img_height']):
        offset_y = data['img_height'] - center[1]
    if (center[1] - offset_y < 0):
        offset_y = center[1]
    return (offset_x, offset_y)

def visualiseImage(image, bbox, center, joints):
    img = image.copy()
    img_croppad = image.copy()
    img_croppad = img_croppad[center[1]-bbox[1]:center[1]+bbox[1], center[0]-bbox[0]:center[0]+bbox[0]]
    
    cv2.line(img, (center[0]-bbox[0],center[1]-bbox[1]), (center[0]+bbox[0],center[1]-bbox[1]), (255, 0, 0), 2)
    cv2.line(img, (center[0]+bbox[0],center[1]-bbox[1]), (center[0]+bbox[0],center[1]+bbox[1]), (255, 0, 0), 2)
    cv2.line(img, (center[0]+bbox[0],center[1]+bbox[1]), (center[0]-bbox[0],center[1]+bbox[1]), (255, 0, 0), 2)
    cv2.line(img, (center[0]-bbox[0],center[1]+bbox[1]), (center[0]-bbox[0],center[1]-bbox[1]), (255, 0, 0), 2)
   
    for j in range(len(joints)):
        cv2.circle(img_croppad, (int(joints[j][0]), int(joints[j][1])), 3, (0, 255, 255), -1)
    
    cv2.imshow('Selected image',img)
    cv2.imshow('Cropped image',img_croppad)
    cv2.waitKey()
       
def generateGaussian(pos, mean, Sigma):
    rv = multivariate_normal([mean[1],mean[0]], Sigma)
    tmp = rv.pdf(pos)
    hmap = np.multiply(tmp, np.sqrt(np.power(2*np.pi,2)*np.linalg.det(Sigma)))
    return hmap

def generateHeatMaps(center, joints):
    num_joints = len(joints_idx)
    heatMaps = np.zeros((inputSizeNN,inputSizeNN,num_joints+1))
    sigma_sq = np.power(sigma,2)
    Sigma = [[sigma_sq,0],[0,sigma_sq]]
    
    x, y = np.mgrid[0:368, 0:368]
    pos = np.dstack((x, y))
    
    # heatmaps representing the position of the joints
    for i in range(num_joints):
        heatMaps[:,:,i] = generateGaussian(pos, joints[i], Sigma)
        heatMaps[:,:,-1] = np.sum([heatMaps[:,:,-1], heatMaps[:,:,i]], axis=0)
    cv2.imshow('hm',heatMaps[:,:,-1])
    
    # heatmap to be added to the RGB image
    center = generateGaussian(pos, center, Sigma)
    return heatMaps, center


def runCaffeOnModel(data, model_dir, idx):
    iterNumber = getIter(model_dir)
    print '\n\nEvaluating iteration %d\n\n' % iterNumber
    for i in range(len(idx)):
        fno = idx[i]
        if (not data[fno]['isValidation']):
			continue

        curr_data = data[fno]
        center = map(int, curr_data['objpos'])
        bbox = getBoundingBox(curr_data)
              
        # take data
        img = cv2.imread(curr_data['img_paths'])
        joints = filterJoints(curr_data['joint_self'])
        
        # crop around person
        img_croppad = img[center[1]-bbox[1]:center[1]+bbox[1], center[0]-bbox[0]:center[0]+bbox[0]]
        
        # transform data
        offset_left = - (center[0] - bbox[0])
        offset_top = - (center[1] - bbox[1])
        center = np.sum([center, [offset_left, offset_top]], axis=0)
        for j in range(len(joints)):
            joints[j][0] += offset_left
            joints[j][1] += offset_top
            del joints[j][2]
        
        # visualize data
        if (verbose):
            visualiseImage(img, bbox, map(int, curr_data['objpos']), joints)
        
        # resize image and update joint positions
        resizedImage = cv2.resize(img_croppad, (inputSizeNN,inputSizeNN))
        fx = inputSizeNN/img_croppad.shape[1]
        fy = inputSizeNN/img_croppad.shape[0]
        center = map(int, np.multiply(center, [fx,fy]))
        for j in range(len(joints)):
            joints[j] = map(int, np.multiply(joints[j], [fx,fy]))
        
        if (verbose):
            tmp = resizedImage.copy()
            for j in range(len(joints)):
                cv2.circle(resizedImage, (joints[j][0], joints[j][1]), 3, (0, 255, 255), -1)
            cv2.circle(resizedImage, (center[0], center[1]), 3, (255, 255, 255), -1)
            cv2.imshow('final res', tmp)
            cv2.waitKey()
        
        heatMaps = generateHeatMaps(center, joints)
        # TODO: check both same type (uint8 or float) - heatmaps and nn's output
        # TODO: convert the image from w x h x c into c h w x h using img4ch = np.transpose(img4ch, (2, 0, 1))
        print '1'
    
def combine_data(val, new_val):
    for i in range(len(new_val['iteration'])):
        val['iteration'].append(new_val['iteration'][i])
        val['loss_iter'].append(new_val['loss_iter'][i])
        val['loss_stage'].append(new_val['loss_stage'][i])
        val['stage'].append(new_val['stage'][i])
    return val

def getIter(item):
    regex_iteration = re.compile('pose_iter_(\d+).caffemodel')
    iter_match = regex_iteration.search(item)
    return int(iter_match.group(1))

def getLossOnValidationSet(json_file, models):
    files = [f for f in os.listdir(models) if f.endswith('.caffemodel')]
    files = sorted(files, key=getIter)
    val = dict([('iteration',[]), ('loss_iter',[]), ('loss_stage',[]), ('stage',[])])
    
    with open(json_file) as data_file:
        data_this = json.load(data_file)
        data_this = data_this['root']
        data = data_this
        
    numSample = len(data)
    print 'overall data %d' % len(data)
    idx = range(0, numSample, samplingRate)
    
    for i in range(len(files)):
        model_dir = '%s/%s' % (models, files[i])
        new_val = runCaffeOnModel(data, model_dir, idx)
        val = combine_data(val, new_val)
    return val

def main():
    caffe_dir = os.environ.get('CAFFE_HOME_CPM')
    #lmdb_dir = '%s/models/cpm_architecture/lmdb/val_small' % caffe_dir
    json_file = '%s/models/cpm_architecture/jsonDatasets/H36M_annotations.json' % caffe_dir
    caffe_models_dir = '%s/models/cpm_architecture/prototxt/caffemodel/trial_1/' % caffe_dir
    
    loss = getLossOnValidationSet(json_file, caffe_models_dir)
    
    # TODO: save file ro be read by the readLofFile.py script
    

if __name__ == '__main__':
    main()
