# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

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

# Intel Caffe
This fork is dedicated to improving Caffe performance when running on CPU (in particular Xeon servers)

## Performance Results :
Time measures are: average Forward-Backward as stated by *caffe time*. *speedup* is (bvlc-caffe-master branch measure) / (intelcaffe-master branch measure)

#### Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz (MKL 11.3, GCC 4.8.3):
Branch | googlenet(speedup: 6.5) | caffenet (speedup: 6.4) | alexnet(speedup: 6.2) | cifar10-sigmoid-bn(speedup: 9.5) 
----------|-----------------------|-------------------------------|-----------------------------|---------------------
intelcaffe-master |682ms|1276ms|1387ms|34ms
bvlc-caffe-master |4438ms|8164ms|8644ms |323ms

#### Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz (OpenBLAS 0.2.14, GCC 4.8.3):
Branch | googlenet(speedup: 17.0) | caffenet (speedup: 8.7) | alexnet(speedup: 15.4)| ciphar10-sigmoid-bn(speedup: 8.4) 
----------|-----------------------|-------------------------------|----------|-------------------
intelcaffe-openmp |1169ms|3088ms|4628ms|63ms  
bvlc-caffe-master |19767|26993ms|71152ms|529ms

So there is significant speedup, that depends on how many CPU cores the platform does have. Tests were made using MKL(it is available free of charge now) and OpenBLAS.

- **It is best to have HT disabled in BIOS as using OMP_NUM_THREADS may not always prevent
OS kernel for running two OpenMP threads on the same phyisical core (eg. using HT).** 
- Without using OMP_NUM_THREADS and with HT enabled performance is still better than single threaded version (data not included here)

### Building:
Build as usual, either from makefile or cmake. Both build systems will detect if openmp is available for compiler of your choice and use it

### Running:
It is best NOT to use Hyperthreading . So either disable it in BIOS or limit OpenMP threads number by using OMP_NUM_THREADS env variable. If not sure how to set OMP_NUM_THREADS  and you cannot disable HT in BIOS, then do not do it, you should still observe performance gain , but not that
significant as when not relying on HT.

##### Example of running:
###### Intel(R) Xeon(R) E5-2699 v3 @ 2.30GHz, two sockets, 18 cpu cores in each socket
* GOMP_CPU_AFFINITY="0-35" OMP_PROC_BIND=false OMP_NUM_THREADS=36 ./build/tools/caffe time -iterations 50  --model=models/bvlc_googlenet/train_val.prototxt*  

##### Notes:
To check if you have HT enabled:
*sudo dmidecode -t processor | grep HTT*
or 
*cat /proc/cpuinfo | grep ht*

With HT (usually) You have twice as much processors online that physical cpu cores. So get this number of processors online , divide by 2 and make this result a value to which
 OMP_NUM_THREADS will be set to.

If you have any questions, please do not hesitate to ask.


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
