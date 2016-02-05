# Caffe

[![Travis Build Status](https://travis-ci.org/pavlejosipovic/caffe.svg?branch=master)](https://travis-ci.org/pavlejosipovic/caffe) Travis (Linux build)

[![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/oagoige3a0kdg4dp/branch/master?svg=true)] (https://ci.appveyor.com/project/pavlejosipovic/caffe) AppVeyor (Windows build)

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

## Windows Setup
**Requirements**: Visual Studio 2013

By defaults Windows build requires `CUDA` and `cuDNN` libraries.
Both can be disabled by adjusting build variables in `.\windows\CommonSettings.props`.
Python support is disabled by default, but can be enabled via `.\windows\CommonSettings.props` as well.
3rd party dependencies required by Caffe are automatically resolved via NuGet.

### CUDA
Download `CUDA Toolkit 7.5` [from nVidia website](https://developer.nvidia.com/cuda-toolkit).
If you don't have CUDA installed, you can experiment with CPU_ONLY build.
In `.\windows\CommonSettings.props` set `CpuOnlyBuild` to `true` and set `UseCuDNN` to `false`.

### cuDNN
Download `cuDNN v3` or `cuDNN v4` [from nVidia website](https://developer.nvidia.com/cudnn).
Unpack downloaded zip to `CuDnnPath` defined in `.\windows\CommonSettings.props`.
Alternatively, you can disable cuDNN by setting `UseCuDNN` to `false` in the property file.

### Python
To build Caffe Python wrapper set `PythonSupport` to `true` in `.\windows\CommonSettings.props`.
Download Miniconda 2.7 64-bit Windows installer [from Miniconda website] (http://conda.pydata.org/miniconda.html).
Install for all users and add Python to PATH (through installer).

Run the following commands from elevated command prompt:

```
conda install --yes numpy scipy matplotlib scikit-image pip
pip install protobuf
```

### Build
Now, you should be able to build `.\windows\Caffe.sln`

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
