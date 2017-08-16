---
title: MobileNet-SSD examples
description: simple case to demo MobileNet-SSD and test the accuracy
category: example
---

## Video detection examples.
There is a brief video detection demo to show the capacity of MobileNet-SSD. please follow below steps to setup
+ Download the pretrained weights `MobileNetSSD_deploy.caffemodel`on https://drive.google.com/file/d/0B3gersZ2cHIxRm5PMWRoTkdHdHc/view
+ On the `$CAFFE_ROOT` folder, run command below, it will display the detection video on the screen and save the output video as test.avi
```
./build/tools/caffe test -model examples/mobilenet_ssd/MobileNetSSD_video_example.prototxt -weights $WEIGHTSPATH/MobileNetSSD_deploy.caffemodel -gpu 0 -phase TEST -iterations 10000
```
+ You can use your own video by modify the `video_file` parameter on the two `VideoData` layer 
## Verify the accuracy of detection network.
caffe/caffe-fp16 tool of clCaffe support the measurement of mAP of detection network. To verify the accuracy of MobileNet-SSD on VOC database, please follow below steps. The float and half version of clCaffe can achieve 0.727 mAP for VOC2007 test data.
+ Downlad VOC2007 and VOC2012 dataset
```
# Download the data.
cd $HOME/data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
# Extract the data.
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
```
+ Create the LMDB file.
```
cd $CAFFE_ROOT
# Create the trainval.txt, test.txt, and test_name_size.txt in data/VOC0712/
./data/VOC0712/create_list.sh
# You can modify the parameters in create_data.sh if needed.
# It will create lmdb files for trainval and test with encoded original image:
#   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb
#   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb
# and make soft links at examples/VOC0712/
./data/VOC0712/create_data.sh
```
+ Verify the mAP on $CAFFE_ROOT directory with below command. Consider the `iterations` here is 619, that's because the `batch` we use is `8` and the total images of VOC test database is `8*619=4952`. The weights used here can be download from google driver link provided on the previous section.
```
./build/tools/caffe test -detection -model examples/mobilenet_ssd/MobileNetSSD_test.prototxt -weights $WEIGHTPATH/MobileNetSSD_deploy.caffemodel -gpu 0 -phase TEST -iterations 619
```
+ Wait in patience to see the detecion_eval output on the terminal. You can also test your own data by prepared your own `lmdb` database and `labelmap` portotxt.
