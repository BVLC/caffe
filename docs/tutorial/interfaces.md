---
title: Interfaces
---
# Interfaces

Caffe has command line, Python, and MATLAB interfaces for day-to-day usage, interfacing with research code, and rapid prototyping. While Caffe is a C++ library at heart and it exposes a modular interface for development, not every occasion calls for custom compilation. The cmdcaffe, pycaffe, and matcaffe interfaces are here for you.

## Command Line

The command line interface -- cmdcaffe -- is the `caffe` tool for model training, scoring, and diagnostics. Run `caffe` without any arguments for help. This tool and others are found in caffe/build/tools. (The following example calls require completing the LeNet / MNIST example first.)

**Training**: `caffe train` learns models from scratch, resumes learning from saved snapshots, and fine-tunes models to new data and tasks:

* All training requires a solver configuration through the `-solver solver.prototxt` argument. 
* Resuming requires the `-snapshot model_iter_1000.solverstate` argument to load the solver snapshot. 
* Fine-tuning requires the `-weights model.caffemodel` argument for the model initialization.

For example, you can run:

    # train LeNet
    caffe train -solver examples/mnist/lenet_solver.prototxt
    # train on GPU 2
    caffe train -solver examples/mnist/lenet_solver.prototxt -gpu 2
    # resume training from the half-way point snapshot
    caffe train -solver examples/mnist/lenet_solver.prototxt -snapshot examples/mnist/lenet_iter_5000.solverstate

For a full example of fine-tuning, see examples/finetuning_on_flickr_style, but the training call alone is

    # fine-tune CaffeNet model weights for style recognition
    caffe train -solver examples/finetuning_on_flickr_style/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel

**Testing**: `caffe test` scores models by running them in the test phase and reports the net output as its score. The net architecture must be properly defined to output an accuracy measure or loss as its output. The per-batch score is reported and then the grand average is reported last.

    #
    # score the learned LeNet model on the validation set as defined in the 
    # model architeture lenet_train_test.prototxt
    caffe test -model examples/mnist/lenet_train_test.prototxt -weights examples/mnist/lenet_iter_10000.caffemodel -gpu 0 -iterations 100

**Benchmarking**: `caffe time` benchmarks model execution layer-by-layer through timing and synchronization. This is useful to check system performance and measure relative execution times for models.

    # (These example calls require you complete the LeNet / MNIST example first.)
    # time LeNet training on CPU for 10 iterations
    caffe time -model examples/mnist/lenet_train_test.prototxt -iterations 10
    # time LeNet training on GPU for the default 50 iterations
    caffe time -model examples/mnist/lenet_train_test.prototxt -gpu 0
    # time a model architecture with the given weights on the first GPU for 10 iterations
    caffe time -model examples/mnist/lenet_train_test.prototxt -weights examples/mnist/lenet_iter_10000.caffemodel -gpu 0 -iterations 10

**Diagnostics**: `caffe device_query` reports GPU details for reference and checking device ordinals for running on a given device in multi-GPU machines.

    # query the first device
    caffe device_query -gpu 0

## Python

The Python interface -- pycaffe -- is the `caffe` module and its scripts in caffe/python. `import caffe` to load models, do forward and backward, handle IO, visualize networks, and even instrument model solving. All model data, derivatives, and parameters are exposed for reading and writing.

- `caffe.Net` is the central interface for loading, configuring, and running models. `caffe.Classsifier` and `caffe.Detector` provide convenience interfaces for common tasks.
- `caffe.SGDSolver` exposes the solving interface.
- `caffe.io` handles input / output with preprocessing and protocol buffers.
- `caffe.draw` visualizes network architectures.
- Caffe blobs are exposed as numpy ndarrays for ease-of-use and efficiency.

Tutorial IPython notebooks are found in caffe/examples: do `ipython notebook caffe/examples` to try them. For developer reference docstrings can be found throughout the code.

Compile pycaffe by `make pycaffe`. The module dir caffe/python/caffe should be installed in your PYTHONPATH for `import caffe`.

## MATLAB

The MATLAB interface -- matcaffe -- is the `caffe` mex and its helper m-files in caffe/matlab. Load models, do forward and backward, extract output and read-only model weights, and load the binaryproto format mean as a matrix.

A MATLAB demo is in caffe/matlab/caffe/matcaffe_demo.m

Note that MATLAB matrices and memory are in column-major layout counter to Caffe's row-major layout! Double-check your work accordingly.

Compile matcaffe by `make matcaffe`.
