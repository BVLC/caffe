# A No Reference Image Quality Assessment System

This project implements a no-reference image quality assessment convolutional neural network (CNN) using the deep learning framework Caffe.
It essentially implements the paper cited in the citations section below.
"Training data preparation scripts" directory contains MATLAB scripts for data pre-processing and preparation.
compute_LCC_SROCC.m is a MATLAB script which computes the linear and Pearson correlation coefiicients of the final results.
train_IQA_CNN_gpu_x.sh scripts train the network using the GPU id x on mulit-GPU machines. When running on a machine with only single GPU use train_IQA_CNN_gpu_0.sh script in the project root directory like this:
./train_IQA_CNN_gpu_0.sh 1 10 fastfading
This will train 10 networks using different combinations of training and validation data subsets on fastfading distortion type. To train the network on all distortions combined use "all" (without quotes) as the third argument to the script


## Dataset used
The network is trained and tested using LIVE database. Using all five distortion types in the database

## License and Citations

If you find this work useful in your projects or research then please cite this project (https://github.com/Adnan1011/NR-IQA-CNN)

NR IQA CNN paper:
    
    @article{The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2014, pp.1733–1740,
      Author = {Le Kang, Peng Ye, Yi Li and David Doermann},
      Conference = {CVPR 2014},
      Title = {Convolutional Neural Networks for No-Reference Image Quality Assessment},
      Year = {2014}
    }

This repository was initially forked from Caffe repository.
Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
