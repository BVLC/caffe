__author__ = 'pittnuts'
import sys
import caffe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import argparse
from scipy import misc
import cv2
from PIL import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    args = parser.parse_args()

    h_target = args.height
    w_target = args.width
    assert h_target>0 and w_target>0
    r_target = (h_target/(float)(w_target))

    if not os.path.isdir(args.input_dir):
        raise IOError("input director {} does not exist".format(args.input_dir))
    if not os.path.isdir(args.output_dir):
        print "creating output directory: {}".format(args.output_dir)
        os.mkdir(args.output_dir)

    files_dirs = os.listdir( args.input_dir )
    print "Totally {} files".format(files_dirs.__len__())
    count = 0
    for fd in files_dirs:
        file = args.input_dir+'/'+fd
        if os.path.isfile(file):
            try:
                img = Image.open(file)#misc.imread(file) #cv2.imread(file)#
                #plt.imshow(img)
                #plt.show()

                h_orig = img.size[1]#img.shape[0] #
                w_orig = img.size[0]#img.shape[1] #
                r_orig = (h_orig/(float)(w_orig))
                h_crop = 0
                w_crop = 0
                if r_target < r_orig:
                    h_crop = int(round((h_orig - w_orig*r_target)/2.0))
                else:
                    w_crop = int(round((w_orig - h_orig/r_target)/2.0))

                #crop_area = img[h_crop:h_orig-h_crop,w_crop:w_orig-w_crop]
                crop_area = img.crop((w_crop,h_crop,w_orig-w_crop,h_orig-h_crop))
                img_target = crop_area.resize((w_target,h_target),Image.ANTIALIAS)#cv2.resize(crop_area,(h_target,w_target),interpolation=cv2.INTER_CUBIC)#
                if  "ILSVRC2012_val_00035099.JPEG".__eq__(fd):
                #    plt.imshow(img_target)
                #    #img_target.show()
                #    plt.show()
                    pass
                count += 1
                if count%1000==0:
                    print "{} files are resized".format(count)
                #assert img_target.shape[0]==h_target and img_target.shape[1]==w_target
                #misc.imsave(args.output_dir+'/'+fd, img_target)
                #plt.imsave(args.output_dir+'/'+fd, img_target)
                #cv2.imwrite(args.output_dir+'/'+fd,img_target)
                img_target.save(args.output_dir+'/'+fd)
                img_check = Image.open(args.output_dir+'/'+fd)
                if (img_check.size[1]!=h_target) or (img_target.size[0]!=w_target):
                    print "{} {}x{}(hxw) resizing failed".format(fd,img_check.size[1],img_check.size[0])

            except IOError as e:
                print "I/O warning({0}): {1}. Ignore resizing {2}".format(e.errno, e.strerror, file)


    print "Done! {} files are resized".format(count)