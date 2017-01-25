'''
Plot the detection results output by ssd_detect.cpp.
'''

import argparse
from collections import OrderedDict
from google.protobuf import text_format
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as io
import sys

import caffe
from caffe.proto import caffe_pb2

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def showResults(img_file, results, labelmap=None, threshold=None, display=None):
    if not os.path.exists(img_file):
        print "{} does not exist".format(img_file)
        return
    img = io.imread(img_file)
    plt.clf()
    plt.imshow(img)
    plt.axis('off');
    ax = plt.gca()
    if labelmap:
        # generate same number of colors as classes in labelmap.
        num_classes = len(labelmap.item)
    else:
        # generate 20 colors.
        num_classes = 20
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    for res in results:
        if 'score' in res and threshold and float(res["score"]) < threshold:
            continue
        label = res['label']
        name = "class " + str(label)
        if labelmap:
            name = get_labelname(labelmap, label)[0]
        if display_classes and name not in display_classes:
            continue
        color = colors[label % num_classes]
        bbox = res['bbox']
        coords = (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1]
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=3))
        if 'score' in res:
            score = res['score']
            display_text = '%s: %.2f' % (name, score)
        else:
            display_text = name
        ax.text(bbox[0], bbox[1], display_text, bbox={'facecolor':color, 'alpha':0.5})
    if len(results) > 0 and "out_file" in results[0]:
        plt.savefig(results[0]["out_file"], bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "Plot the detection results output by ssd_detect.")
    parser.add_argument("resultfile",
            help = "A file which contains all the detection results.")
    parser.add_argument("imgdir",
            help = "A directory which contains the images.")
    parser.add_argument("--labelmap-file", default="",
            help = "A file which contains the LabelMap.")
    parser.add_argument("--visualize-threshold", default=0.01, type=float,
            help = "Display detections with score higher than the threshold.")
    parser.add_argument("--save-dir", default="",
            help = "A directory which saves the image with detection results.")
    parser.add_argument("--display-classes", default=None,
            help = "If provided, only display specified class. Separate by ','")

    args = parser.parse_args()
    result_file = args.resultfile
    img_dir = args.imgdir
    if not os.path.exists(img_dir):
        print "{} does not exist".format(img_dir)
        sys.exit()
    labelmap_file = args.labelmap_file
    labelmap = None
    if labelmap_file and os.path.exists(labelmap_file):
        file = open(labelmap_file, 'r')
        labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), labelmap)
    visualize_threshold = args.visualize_threshold
    save_dir = args.save_dir
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    display_classes = args.display_classes

    img_results = OrderedDict()
    with open(result_file, "r") as f:
        for line in f.readlines():
            img_name, label, score, xmin, ymin, xmax, ymax = line.strip("\n").split()
            img_file = "{}/{}".format(img_dir, img_name)
            result = dict()
            result["label"] = int(label)
            result["score"] = float(score)
            result["bbox"] = [float(xmin), float(ymin), float(xmax), float(ymax)]
            if save_dir:
                out_file = "{}/{}.png".format(save_dir, os.path.basename(img_name))
                result["out_file"] = out_file
            if img_file not in img_results:
                img_results[img_file] = [result]
            else:
                img_results[img_file].append(result)
    for img_file, results in img_results.iteritems():
        showResults(img_file, results, labelmap, visualize_threshold, display_classes)
