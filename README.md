Note : This repo is not aimed to be merged with Caffe, and so is being maintained as seperate repo, instead of a fork. 

This Repo is aimed to enhance Caffe to include layers that are required to construct Extreme Learning Machine. Currently, only Least square layer is available for constructing ELM (by combining Inner Product layer and Sigmoid Layer). Iterative Least square support is under development to make ELM Online Sequential. 

Additionally Transpose Layer is provided with this repo to make the construction of stacked ELM-Auto Encoders possible.

####LS Layer
- Bottom : "data"
- Bottom : "labels"
- Param{Name : "beta"}
- //no top 
- beta (ß) is the weight calculated as Least square solution of Hß = Y, where H is "data" or bottom[0] and Y is "labels" or bottom[1].
- Requires Intel MKL Library


####Transpose Layer
- //no bottom
- Param{Name : "beta"}
- Param{Name : "transposed_beta"}
- //no top
- Currently requires Intel MKL Library, but will soon be updated to work without it.


####Other changes includes:
- addition of some functions to math_functions.cpp and hpp. 
- some changes to net.cpp so that, transpose layer can be setup to share weights of any layer without knowing the size of blob.

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
