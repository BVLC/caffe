# Deep Face Recognition with Caffe Implementation

This branch is developed for deep face recognition, the related paper is as follows.
    
    A Discriminative Feature Learning Approach for Deep Face Recognition[C]
    Yandong Wen, Kaipeng Zhang, Zhifeng Li*, Yu Qiao
    European Conference on Computer Vision. Springer International Publishing, 2016: 499-515.


* [Updates](#updates)
* [Files](#files)
* [Train_Model](#train_model)
* [Extract_DeepFeature](#extract_deepfeature)
* [Contact](#contact)
* [Citation](#citation)
* [LICENSE](#license)
* [README_Caffe](#readme_caffe)

### Updates
- Oct 13, 2016
  * A demo for extracting deep feature by the given model is provided.
- Oct 12, 2016
  * The links of face model and features on LFW are available.   
  **model:** [google drive](https://drive.google.com/open?id=0B_geeR2lTMegUzlSdG5wZ1V5WU0) [baidu skydrive](http://pan.baidu.com/s/1skFoqrr)  
  **feature:** [google drive](https://drive.google.com/open?id=0B_geeR2lTMegLWRuWnZoMVJPZ3c) [baidu skydrive](http://pan.baidu.com/s/1boLM1bh)
  * The training prototxt of toy example on MNIST are released.
- Otc 9, 2016
  * The code and training prototxt for our [ECCV16](http://link.springer.com/chapter/10.1007/978-3-319-46478-7_31) paper are released. 
  * If you train our Network on **CAISA-WebFace**, the expected verification performance of **SINGLE MODEL** on **[LFW](http://vis-www.cs.umass.edu/lfw/)** should be **~99%**.

### Files
- Original Caffe library
- Center Loss
  * src/caffe/proto/caffe.proto
  * include/caffe/layers/center_loss_layer.hpp
  * src/caffe/layers/center_loss_layer.cpp
  * src/caffe/layers/center_loss_layer.cu
- face_example
  * face_example/data/
  * face_example/face_snapshot/
  * face_example/face_train_test.prototxt
  * face_example/face_solver.prototxt
  * face_example/face_deploy.prototxt
  * face_example/extractDeepFeature.m
- mnist_example
  * mnist_example/data/
  * mnist_example/face_snapshot/
  * mnist_example/mnist_train_test.prototxt
  * mnist_example/mnist_solver.prototxt
  * mnist_example/mnist_deploy.prototxt

### Train_Model
1. The Installation completely the same as [Caffe](http://caffe.berkeleyvision.org/). Please follow the [installation instructions](http://caffe.berkeleyvision.org/installation.html). Make sure you have correctly installed before using our code. 
2. Download the face dataset for training, e.g. [CAISA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html), [VGG-Face](http://www.robots.ox.ac.uk/~vgg/data/vgg_face/), [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/), [MegaFace](http://megaface.cs.washington.edu/).
3. Preprocess the training face images, including detection, alignment, etc. Here we strongly recommend [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment), which is an effective and efficient open-source tool for face detection and alignment.
4. Creat list for training set and validation set. Place them in face_example/data/
5. Specify your data source for train & val

        layer {
          name: "data"
          type: "ImageData"
          top: "data"
          top: "label"
          image_data_param {
            source: "face_example/data/###your_list###"
          }
        }

6. Specify the number of subject in FC6 layer

        layer {
          name: "fc6"
          type: "InnerProduct"
          bottom: "fc5"
          top: "fc6"
          inner_product_param {
            num_output: ##number##
          }
        }

7. Specify the loss weight and the number of subject in center loss layer

        layer {
          name: "center_loss"
          type: "CenterLoss"
          bottom: "fc5"
          bottom: "label"
          top: "center_loss"
          loss_weight: ##weight##
          center_loss_param {
            num_output: ##number##
          }
        }

8. Train model

        cd $CAFFE-FACE_ROOT
        ./build/tools/caffe train -solver face_example/face_solver.prototxt -gpu X,Y

### Extract_DeepFeature
1. Compile matcaffe by make matcaffe
2. Specify the correspinding paths in face_example/extractDeepFeature.m

        addpath('path_to_matCaffe/matlab');
        model = 'path_to_deploy/face_deploy.prototxt';
        weights = 'path_to_model/face_model.caffemodel';
        image = imread('path_to_image/Jennifer_Aniston_0016.jpg');

3. Run extractDeepFeature.m in Matlab

### Contact 
- [Yandong Wen](http://ydwen.github.io/)
- [Kaipeng Zhang](http://kpzhang93.github.io/)

### Citation
You are encouraged to cite the following paper if it helps your research. 

    @inproceedings{wen2016discriminative,
      title={A Discriminative Feature Learning Approach for Deep Face Recognition},
      author={Wen, Yandong and Zhang, Kaipeng and Li, Zhifeng and Qiao, Yu},
      booktitle={European Conference on Computer Vision},
      pages={499--515},
      year={2016},
      organization={Springer}
    }

### License
Copyright (c) Yandong Wen

All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
