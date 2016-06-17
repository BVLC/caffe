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

def tmpFunction(x,y):
    center = [183,183]
    d2 = np.power((x-center[1]),2) + np.power((y-center[0]),2)
    exponent = d2/2.0/sigma/sigma
    if exponent > 4.6052:
        return 0.0
    exponent = d2/2.0/sigma/sigma
    if np.exp(-exponent) > 1:
        return 1
    else:
        return np.exp(-exponent)
        
def generateGaussian(hmap, pos):
    for gx in range(inputSizeNN):
        for gy in range(inputSizeNN):
            d2 = np.power((gx-pos[1]),2) + np.power((gy-pos[0]),2)
            exponent = d2/2.0/sigma/sigma
            if exponent > 4.6052:
                continue
            hmap[gx,gy] += np.exp(-exponent)
            if hmap[gx,gy] > 1:
                hmap[gx,gy] = 1
    #np.fromfunction(tmpFunction, (inputSizeNN,inputSizeNN))
    return hmap

def generateHeatMaps(center, joints):
    num_joints = len(joints_idx)
    heatMaps = np.zeros((inputSizeNN,inputSizeNN,num_joints+1))
    
    for i in range(num_joints):
        heatMaps[:,:,i] = generateGaussian(heatMaps[:,:,i], joints[i])
    for i in range(num_joints):
        heatMaps[:,:,-1] = np.sum([heatMaps[:,:,-1], heatMaps[:,:,i]], axis=2)
    cv2.imshow('hm',heatMaps[:,:,-1])
    
#    res = np.random.multivariate_normal(mean, cov, (368,368))
#    res[:,:,0] = np.divide(np.subtract(res[:,:,0], np.min(res[:,:,0])),np.max(res[:,:,0])-np.min(res[:,:,0]))
#    cv2.imshow('hm',res[:,:,0])
    
    return heatMaps

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
