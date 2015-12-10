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

# Intel Caffe (OpenMP branch)

## Performance Results :
Time measures are: average Forward-Backward as stated by *caffe time*. *speedup* is (master branch measure / openmp-conv-relu branch measure)

#### Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz (MKL 11.3, GCC 4.8.3):
Branch | googlenet(speedup: 5.5) | caffenet (speedup: 5.9) | alexnet(speedup: 5.5) | ciphar10-sigmoid-bn(speedup: 7.5) 
----------|-----------------------|-------------------------------|-----------------------------|---------------------
openmp (using OMP_NUM_THREADS=36)| 813ms|1369ms|1547ms|43ms
master |4438ms                    |8164ms|8644ms |323ms

#### Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz (OpenBLAS 0.2.14, GCC 4.8.3):
Branch | googlenet(speedup: 2.4) | caffenet (speedup: 3.7) | alexnet(speedup: 1.1)| ciphar10-sigmoid-bn(speedup: 6.6) 
----------|-----------------------|-------------------------------|-----------------------------
openmp (using OMP_NUM_THREADS=36)| 7033ms|7076ms |57980ms|81ms  
master |16848ms 	|26130ms 	|62091ms|538ms

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
###### Xeon E5 , two sockets, 18 cpu cores in each socket
*OMP_NUM_THREADS=36 ./build/tools/caffe time -iterations 20  --model=models/bvlc_googlenet/train_val.prototxt*  
###### Xeon E3 , one socket, 4 cpu cores in each socket
*OMP_NUM_THREADS=4 ./build/tools/caffe time -iterations 20  --model=models/bvlc_googlenet/train_val.prototxt*   
###### Brix i7-4770R (Haswell), one socket, 4 cpu cores in each socket
*OMP_NUM_THREADS=4 ./build/tools/caffe time -iterations 20  --model=models/bvlc_googlenet/train_val.prototxt*   


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
