# Deep Face Recognition with Caffe Implementation

This branch is developed for deep face recognition, the related paper is as follows.
    
    A Discriminative Feature Learning Approach for Deep Face Recognition[C]
    Yandong Wen, Kaipeng Zhang, Zhifeng Li*, Yu Qiao
    European Conference on Computer Vision. Springer International Publishing, 2016: 499-515.


* [Updates](#updates)
* [Files](#files)
* [Usage](#usage)
* [Contact](#contact)
* [Citation](#citation)
* [LICENSE](#license)
* [README_Caffe](#readme_caffe)


### Updates
- Otc 9, 2016
  * The code and training prototxt for our [ECCV16](http://link.springer.com/chapter/10.1007/978-3-319-46478-7_31) paper are released.

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

### Usage
- Make sure you have correctly installed [Caffe](http://caffe.berkeleyvision.org/) before using our code. Please follow the [installation instructions](http://caffe.berkeleyvision.org/installation.html).
- Download the face dataset for training, e.g. [CAISA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html), [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/), [MegaFace](http://megaface.cs.washington.edu/).
- Preprocess the training face images, including detection, alignment, etc. Here we strongly recommend [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment), which is an effective and efficient open-source tool for face detection and alignment.
- Creat list for training set and validation set. Place them in face_example/data/
- Specify your data source for train & val

        layer {
          name: "data"
          type: "ImageData"
          top: "data"
          top: "label"
          image_data_param {
            source: "face_example/data/###your_list###"
          }
        }
- Specify the number of subject corresponding to your training set

        layer {
          name: "fc6"
          type: "InnerProduct"
          bottom: "fc5"
          top: "fc6"
          inner_product_param {
            num_output: ##number##
          }
        }
- Run the model

        cd $CAFFE-FACE_ROOT
        ./build/tools/caffe train -solver face_example/face_solver.prototxt -gpu X,Y

- Please refer to our ECCV paper and code for more details.

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

***

### README_Caffe
# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }