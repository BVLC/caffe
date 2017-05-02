#!/usr/bin/env bash
SSD_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
$CAFFE_ROOT/build/examples/ssd/ssd_detect examples/ssd/VGGNet/VOC0712/SSD_300x300/deploy.prototxt \
$SSD_ROOT/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel \
$SSD_ROOT/images.txt -out_file detected.txt

python $SSD_ROOT/plot_detections.py \
--labelmap-file ./data/VOC0712/labelmap_voc.prototxt \
detected.txt . --save-dir . --visualize-threshold 0.2

