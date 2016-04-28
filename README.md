# OpenCL Caffe

**This is an experimental, community-maintained branch led by Fabian Tschopp (@naibaf7). It is a work-in-progress.**

**For error reports, please run and include the result of `./build/test/test_all.testbin --gtest_filter=*OpenCLKernelCompileTest* X` where `X` is the OpenCL device to test (i.e. `0`). This test is available after a build with `make all`, `make runtest`.**

This branch of Caffe contains an OpenCL backend and additional layers for fast image segmentation.
This work is partially supported by:
- AMD
- HHMI Janelia
- UZH, INI
- ETH Zurich
- Intel

For a C++ frontend and models to use for image segmentation with this fork, see:
- Frontend: https://github.com/naibaf7/caffe_neural_tool
- Models: https://github.com/naibaf7/caffe_neural_models

## OpenCL Backend

The backend is supposed to work with all vendors. Note however there may be problems with libOpenCL.so provided by nVidia.
It is therefore recommended to install another OpenCL implementation after installing nVidia drivers. Possibilities are:
- Intel OpenCL, see below for details. 
- AMD APP SDK (OpenCL), recommended if you have an AMD GPU or CPU.

### OpenCL for Intel platform for Linux.

For 4th or 5th generation Intel Cores and Intel® Xeon® v3, or Intel® Xeon® v4 processor.
We recommend the driver at the following link: https://software.intel.com/en-us/articles/opencl-drivers#latest_linux_driver.
For 3th generation cores and atom, we recommend Beignet: https://www.freedesktop.org/wiki/Software/Beignet/.

The spatial domain convolution kernel supports all OpenCL platforms now. This convolution kernel
applies auto-tuner mechanism to tune a best kernel for current parameters then store the
result to the subdirectory ".spatialkernels". Thus at the first run, it will take relatively
long time to perform the auto-tuning process. At the second run, it will get the result from the
cache subdirectory directly.

The spatial domain convolution is enabled by default for Intel Gen Graphics paltform. For
other platforms, we need to modify net model specification as below:

add entry "engine: SPATIAL" to all convolution layer specification.

Take AlexNet as an example, we edit file $CAFFE_ROOT/models/bvlc_alexnet/train_val.prototxt, and add the following line to make conv1 layer to be computed using spatial convolution..

<pre><code>
     layer {
       name: "conv1"
       type: "Convolution"
       bottom: "data"
       top: "conv1"
       param {
         lr_mult: 1
         decay_mult: 1
       }
       param {
         lr_mult: 2
         decay_mult: 0
       }
       convolution_param {
         num_output: 96
         kernel_size: 11
         stride: 4
         engine: INTEL_SPATIAL 		<-------------------------- this line!
         weight_filler {
           type: "gaussian"
           std: 0.01
         }
         bias_filler {
           type: "constant"
           value: 0
         }
       }
     }
</code></pre>

To enable the FFT domain convolution, you should install libfftw3, libfftw3f(for cpu) and clfft(for opencl) first.

You can downloaded the fftw3 source code from https://github.com/FFTW/fftw3.git

and the clFFT from https://github.com/listenlink/clFFT.git

Then config the Cmake option with ```-DUSE_FFT=ON``` when using cmake build system or enable the Makefile.config.example line 36 ```USE_FFT := 1``` when using makefile build system

Like the ```INTEL_SPATIAL```, modify the convolution_param to ```engine: FFT```to use fft based convolution engine.

*Please use the latest git master viennacl which has the patch: https://github.com/viennacl/viennacl-dev/pull/181*

## Technical Report
Available on arXiv:
http://arxiv.org/abs/1509.03371

## Further Details

Refer to the BVLC/caffe master branch README for all other details such as license, citation, and so on.
