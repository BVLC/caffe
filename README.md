# SSD: Single Shot MultiBox Detector

By [Wei Liu](http://www.cs.unc.edu/~wliu/), [Dragomir Anguelov](http://research.google.com/pubs/DragomirAnguelov.html), [Dumitru Erhan](http://research.google.com/pubs/DumitruErhan.html), [Christian Szegedy](http://research.google.com/pubs/ChristianSzegedy.html), [Scott Reed](http://www-personal.umich.edu/~reedscot/).

### Introduction

SSD is an unified framework for object detection with a single network. You can use the code to train/evaluate a network for object detection task. For more details, please refer to our [arXiv paper](http://arxiv.org/abs/1512.02325).

<p align="center">
<img src="http://www.cs.unc.edu/~wliu/papers/ssd.png" alt="SSD Framework" width="600px">
</p>

### Citing SSD

Please cite SSD in your publications if it helps your research:

    @article{liu15ssd,
      Title = {{SSD}: Single Shot MultiBox Detector},
      Author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott},
      Journal = {arXiv preprint arXiv:1512.02325},
      Year = {2015}
    }

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Contact](#contact)

:bangbang:**Disclaimer**: This software is still under active development. The result is not the optimal yet.

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/weiliu89/caffe.git
  cd caffe
  git checkout ssd
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  make py
  make test -j8
  make runtest
  # If you have multiple GPUs installed in your machine, make runtest might fail. If so, try following:
  export CUDA_VISIBLE_DEVICES=0; make runtest
  ```

### Preparation
1. Download [fully convolutional reduced (atrous) VGGNet](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6). By default, we assume the model is stored in `$CAFFE_ROOT/models/VGGNet/`

2. Download VOC2007 and VOC2012 dataset. By default, we assume the data is stored in `$HOME/data/`
  ```Shell
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

3. Create the LMDB file.
  ```Shell
  cd $CAFFE_ROOT
  # Create the trainval.txt, test.txt, and test_name_size.txt in data/VOC0712/
  ./data/VOC0712/create_list.sh
  # You can modify the parameters in create_data.sh if needed.
  # It will create lmdb files for trainval and test with encoded original image:
  #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_0x0_0x0_lmdb
  #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_0x0_0x0_lmdb
  # and make soft links at examples/VOC0712/
  ./data/VOC0712/create_data.sh
  ```

### Train/Eval
1. Train your model and evaluate the model on the fly.
  ```Shell
  # It will create model definition files (train.prototxt, test.prototxt, solver.prototxt) and save models in:
  #   - $CAFFE_ROOT/models/VGGNet/VOC0712/SSD_0x0_0x0_0.001/
  # and job file, log file, and the python script in:
  #   - $CAFFE_ROOT/jobs/VGGNet/VOC0712/SSD_0x0_0x0_0.001/
  # and save temporary evaluation results in:
  #   - $HOME/data/VOCdevkit/results/VOC2007/SSD_0x0_0x0_0.001/
  python examples/ssd/ssd_pascal.py
  ```

2. Evaluate the most recent snapshot.
  ```Shell
  # If you would like to test a model you trained, you can do:
  python examples/ssd/score_ssd_pascal.py
  ```

### Contact
Please direct all comments and report all bugs to:

    Wei Liu
    wliu@cs.unc.edu
