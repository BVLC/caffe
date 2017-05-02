---
title: SSD (Single Shot MultiBox Detector) test case
description: simple case to benchmark ssd inference path and simple detector demo.
category: example
---

# Preparation
 1. Download models_VGGNet_VOC0712_SSD_300x300.tar.gz on https://drive.google.com/file/d/0BzKzrI_SkD1_WVVTSmQxU0dVRzA/view
 2. extract and copy folder VGGNet to $CAFFE_ROOT/exampls/ssd/

# Testing
 1. source examples/ssd/ssdvars.sh
 2. run benchmark.sh to benchmark per-layer performance.
 3. run demo.sh to show the simple demo.
